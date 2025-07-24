import requests
import gradio as gr
import os
from typing import Dict, Tuple

# Environment variables
TOKEN = os.environ.get("disease_backend")
API_URL = "https://backend-diseases-priority.hf.space/predict"

# Validate environment variables
if not TOKEN or not API_URL:
    raise ValueError("HF_TOKEN or API_URL not set in environment variables.")

def predict_disease(symptoms: str) -> Dict[str, str]:
    """Predict disease from symptoms using API."""
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"symptoms": symptoms.strip() or ""}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"Server error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Gradio interface
with gr.Blocks(
    title="Health Diagnosis Tool",
    theme=gr.themes.Soft(
        font=["Inter", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="gray"
    )
) as demo:
    gr.Markdown(
        """
        # 🩺 Health Diagnosis Tool
        Enter your symptoms to get a diagnosis and treatment suggestions.
        """,
        elem_classes=["header"]
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            symptom_input = gr.Textbox(
                label="Your Symptoms",
                placeholder="Example: fever, sore throat, fatigue",
                lines=4,
                max_lines=6,
                elem_classes=["symptom-input"]
            )
            submit_btn = gr.Button("Get Diagnosis", variant="primary", elem_classes=["submit-btn"])
            clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["clear-btn"])

        with gr.Column(scale=3):
            with gr.Group():
                diagnosis_output = gr.Textbox(
                    label="Diagnosis",
                    interactive=False,
                    elem_classes=["output-box"]
                )
                treatment_output = gr.Textbox(
                    label="Recommended Treatment",
                    interactive=False,
                    elem_classes=["output-box"]
                )

    def process_symptoms(symptoms: str) -> Tuple[str, str]:
        """Process symptoms and return diagnosis and treatment."""
        if not symptoms.strip():
            return "No symptoms provided", "Please enter symptoms"
        
        result = predict_disease(symptoms)
        if "error" in result:
            return "Error", result["error"]
        
        return (
            result.get("predicted_disease", "Unknown"),
            result.get("treatment", "Consult a healthcare professional")
        )

    def clear_inputs():
        return "", "", ""

    submit_btn.click(
        fn=process_symptoms,
        inputs=symptom_input,
        outputs=[diagnosis_output, treatment_output]
    )

    clear_btn.click(
        fn=clear_inputs,
        inputs=None,
        outputs=[symptom_input, diagnosis_output, treatment_output]
    )

    # Custom CSS
    demo.css = """
    .header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin: 1.5rem 0;
    }
    .symptom-input {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 1rem;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .output-box {
        border-radius: 12px;
        background: #1f2937;
        padding: 1.2rem;
        font-size: 1rem;
        color: #f9fafb;
        margin-bottom: 1rem;
    }
    .submit-btn {
        background: #4f46e5;
        border-radius: 10px;
        padding: 0.8rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .submit-btn:hover {
        background: #4338ca;
    }
    .clear-btn {
        border-radius: 10px;
        padding: 0.8rem;
        font-size: 1rem;
        font-weight: 500;
    }
    .gr-group {
        border-radius: 12px;
        border: 1px solid #4b5563;
        padding: 1rem;
        background: #1f2937;
    }
    """

demo.launch()
