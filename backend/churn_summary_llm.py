from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io, json
import joblib
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

# ---- Load trained RandomForest model ----
model = joblib.load("model_name.pkl")

# ---- Load encoder or feature info ----
try:
    feature_names = joblib.load("encoder_name.pkl")
    if not isinstance(feature_names, list):
        feature_names = []
except:
    feature_names = []

# ---- Configure Gemini ----
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# ---- Initialize sentiment analysis pipeline ----
sentiment_pipeline = pipeline("sentiment-analysis")

# ---- Twilio SMS config ----
account_sid = os.getenv("TWILIO_API_KEY")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(account_sid, auth_token)
twilio_from_number = os.getenv("TWILIO_PHONE_NUMBER")

def send_sms(to_number):
    """Send churn alert SMS to the given number"""
    try:
        message_body = "Dear Customer, we noticed you might churn soon. Contact support for a special offer!"
        msg = twilio_client.messages.create(
            from_=twilio_from_number,
            body=message_body,
            to=to_number
        )
        print(f"SMS sent to {to_number}, SID: {msg.sid}")
    except Exception as e:
        print(f"Failed to send SMS to {to_number}: {e}")
        

def transformers_sentiment_score(text):
    """Convert feedback text to sentiment score"""
    if isinstance(text, str):
        result = sentiment_pipeline(text)[0]
        if result['label'] == 'POSITIVE':
            return result['score']
        elif result['label'] == 'NEGATIVE':
            return -result['score']
        else:
            return 0.0
    else:
        return 0.0


@app.post("/predict")
async def predict_churn(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents),encoding='latin1')
    df_orig = df.copy()

    # ---- Handle FeedbackText → SentimentScore ----
    if "FeedbackText" in df.columns:
        df["SentimentScore"] = df["FeedbackText"].apply(transformers_sentiment_score)
        df = df.drop(columns=["FeedbackText"], errors="ignore")

    # ---- Drop unnecessary columns ----
    drop_cols = ["CustomerID", "Churn","Mobile","Email"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ---- Encode categorical columns ----
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ---- Align columns with training ----
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    # ---- Predict churn ----
    churn_probs = model.predict_proba(df)[:, 1]
    churn_preds = (churn_probs > 0.5).astype(int)

    df_orig["Predicted_Churn"] = churn_preds
    df_orig["Churn_Probability"] = churn_probs.round(2)

    # ---- Send SMS & Email to churned users ----
    churned_customers = df_orig[churn_preds == 1]  # Only churned
    for _, row in churned_customers.iterrows():
        mobile = row.get("Mobile")
        if pd.notna(mobile):
            mobile_str = str(mobile).strip()
            # Prepend +91 if not already present
            if not mobile_str.startswith("+"):
                mobile_str = "+91" + mobile_str

            send_sms(mobile_str)
            
    # ---- 🔹 Compute metrics ----
    total = len(df)
    predicted_churn = int(churn_preds.sum())
    retention_rate = round((1 - predicted_churn / total) * 100, 2)
    avg_prob = round(float(churn_probs.mean()), 2)
    churn_rate = round(predicted_churn / total, 2)
    active_users = total - predicted_churn

    # ---- 🔹 Feature summary ----
    feature_summary = df_orig.describe(include="all").to_string()

    # ---- 🔹 Gemini Insight ----
    prompt = f"""
    You are a data analyst for a subscription-based business.
    Based on these churn metrics and summary, identify which customer segment is at the highest risk of churn.
    Give your response strictly in JSON with one key "High-Risk Segment".

    Metrics:
    - Total Records: {total}
    - Predicted Churn: {predicted_churn}
    - Active Users: {active_users}
    - Retention Rate: {retention_rate}%
    - Churn Rate: {churn_rate}
    - Average Churn Probability: {avg_prob}

    Feature Summary:
    {feature_summary}
    """

    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
    response = model_gemini.generate_content(prompt)

    # ---- 🔹 Parse Gemini Output ----
    text = response.text.strip().replace("```json", "").replace("```", "")
    try:
        insight = json.loads(text)
    except:
        insight = {"High-Risk Segment": text}

    # ---- 🔹 Return JSON result ----
    result = {
        "Total Records": total,
        "Predicted Churn": predicted_churn,
        "Active Users": active_users,
        "Retention Rate": f"{retention_rate}%",
        "Churn Rate": churn_rate,
        "Average Churn Probability": avg_prob,
        "High-Risk Segment": insight.get("High-Risk Segment", "N/A")
    }

    return {"Summary": result}