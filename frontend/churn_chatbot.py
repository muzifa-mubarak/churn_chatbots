import gradio as gr
import pandas as pd
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ---- Load environment variables ----
load_dotenv()

# ---- Configure Gemini API Key ----
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# ---- Use your FastAPI backend URL (Render will call this endpoint) ----
# For local testing: API_URL = "http://127.0.0.1:8000/predict"
API_URL = os.getenv("API_URL", "https://churn-summary-llm.onrender.com/predict")

# ---- Global variable to store uploaded CSV ----
csv_context = None

# ---- Initialize Gemini model ----
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat(history=[])

# ---- Analyze CSV ----
def analyze_csv(file):
    global csv_context
    try:
        df = pd.read_csv(file.name, encoding='latin1')
        csv_context = df.to_string()

        # Send CSV to backend API
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "text/csv")}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json().get("Summary", {})
            total = str(result.get("Total Records", "N/A"))
            churned = str(result.get("Predicted Churn", "N/A"))
            avg_prob = str(result.get("Average Churn Probability", "N/A"))
            retention = str(result.get("Retention Rate", "N/A"))
            high_risk = str(result.get("High-Risk Segment", "N/A"))

            summary_html = f"""
            <div style='background-color:#2b2b2b; padding:20px; border-radius:15px;'>
                <h3 style='color:#f5f5f5;'>📊 Churn Analysis Summary</h3>
                <p><b>Total Customers:</b> {total}</p>
                <p><b>Predicted to Churn:</b> {churned}</p>
                <p><b>Average Churn Probability:</b> {avg_prob}</p>
                <p><b>Retention Rate:</b> {retention}</p>
                <p><b>High-Risk Segment:</b> {high_risk}</p>
            </div>
            """
            return summary_html
        else:
            return f"<p style='color:red;'>❌ API Error: {response.status_code}</p>"

    except Exception as e:
        return f"<p style='color:red;'>⚠️ Failed: {str(e)}</p>"

# ---- Chatbot ----
def chatbot_response(message, history):
    global csv_context
    if csv_context is None:
        return "⚠️ Please upload a CSV first!"

    prompt = f"""
    You are a Churn Analytics Assistant.
    Use the following CSV data context to answer the user’s question:

    {csv_context}

    User Query: {message}
    """

    response = chat.send_message(prompt)
    return response.text

# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as ui:
    gr.Markdown("## 💬 ChurnGuard — Smart Churn Insights & Chatbot")

    with gr.Tab("📈 Analyze Data"):
        with gr.Row():
            csv_input = gr.File(label="Upload Customer Data (CSV)")
            analyze_btn = gr.Button("🔍 Analyze")
        summary_output = gr.HTML(label="Insights")
        analyze_btn.click(analyze_csv, inputs=csv_input, outputs=summary_output)

    with gr.Tab("💬 Chat with Your Data"):
        chatbot = gr.ChatInterface(
            fn=chatbot_response,
            title="Ask ChurnGuard",
            textbox=gr.Textbox(placeholder="Ask about churn patterns, customer insights...", lines=1),
            type="messages"  # Fixes Gradio's deprecation warning
        )

# ---- Render-compatible launch ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    ui.launch(server_name="0.0.0.0", server_port=port)


