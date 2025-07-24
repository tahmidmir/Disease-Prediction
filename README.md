<p align="center">
  <img src="https://via.placeholder.com/150" alt="Disease Prediction API Logo" width="150"/>
</p>

# 🩺 **Disease Prediction API**

This project provides an API built using **FastAPI** that predicts diseases based on user-reported symptoms. It leverages language models like **ClinicalBERT** and **BioGPT** with **RAG (Retrieval-Augmented Generation)** to enhance symptom input, predict the most likely disease, and suggest initial treatments.

## ✨ **Features**
- Accepts raw symptom descriptions from the user
- Enhances symptoms using RAG techniques
- Predicts diseases using trained NLP models
- Returns suggested treatments
- Built with **FastAPI**
- Easily deployable using **Docker**

## 📦 **Installation & Running**

### 🖥️ **Backend**

#### 🛠️ **Build the Docker image**
```bash
docker build -t disease-predictor .

🚀 Run the container
docker run -p 8000:8000 disease-predictor

🔗 API Usage
POST /predict
📥 Input:
{
  "symptoms": "fever, sore throat, fatigue"
}

📤 Response:
{
  "enhanced_symptoms": "...",
  "predicted_disease": "Influenza",
  "treatment": "Rest, hydration, and over-the-counter medication"
}

🛠️ Tech Stack

Python 3.10+
FastAPI
PyTorch
HuggingFace Transformers
ClinicalBERT & BioGPT
Docker

🌐 Gradio Frontend (UI)
The gradio.py file provides a beautiful and responsive web interface using Gradio.
🔑 Key Features:

Free-form symptom input (e.g., fever, cough, fatigue)
"Get Diagnosis" button to send data to the backend API
Display of disease prediction and treatment suggestion
"Clear" button to reset the input/output
Custom CSS styling for a polished UI look

⚙️ How it Works:

User enters symptoms and clicks Get Diagnosis.
Gradio sends a POST request to the FastAPI /predict endpoint.
The backend responds with predicted_disease and treatment.
Results are displayed in the UI.

▶️ Run Gradio:
export disease_backend=your_api_token
python gradio.py

🧠 Models & Technologies Used

ClinicalBERT: Medical text embeddings
BioGPT: Enhanced medical language understanding
RAG (Retrieval-Augmented Generation): Enriches symptom context
FastAPI: REST API backend
Gradio: Frontend interface


