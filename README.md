# Disease Prediction API

This project gives an API constructed using FastAPI that determines diseases on the basis of the symptoms reported by the user. It applies language models such as ClinicalBERT and BioGPT with RAG (Retrieval-Augmented Generation) to improve its input symptoms, recipe of most probable disease, and initial treatment suggestions.


# ✨Features
Accepts raw symptom descriptions from the user

Enhances symptoms using RAG techniques

Predicts diseases using trained NLP models

Returns suggested treatments

Built with FastAPI

Easily deployable using Docker

# 📦Installation & Running

## 🛠️ Backend


### Build the Docker image
```python
docker build -t disease-predictor .
```
### Run the container
```python
docker run -p 8000:8000 disease-predictor
```
### 🌐API Usage
POST /predict
#### 📥Input:

```json
{
  "symptoms": "fever, sore throat, fatigue"
}
```
### 📤 Response:

```json
{
  "enhanced_symptoms": "...",
  "predicted_disease": "Influenza",
  "treatment": "Rest, hydration, and over-the-counter medication"
}
```
### 🧰Tech Stack
Python 3.10+

FastAPI

PyTorch

HuggingFace Transformers

ClinicalBERT & BioGPT

Docker


## 🖥️Gradio Frontend (UI)
The gradio.py file provides a beautiful and responsive web interface using Gradio.

### 🔑Key Features:
Free-form symptom input (e.g., fever, cough, fatigue)

"Get Diagnosis" button to send data to the backend API

Display of disease prediction and treatment suggestion

"Clear" button to reset the input/output

Custom CSS styling for a polished UI look

### ⚙️ How it Works:
User enters symptoms and clicks Get Diagnosis.

Gradio sends a POST request to the FastAPI /predict endpoint.

The backend responds with predicted_disease and treatment.

These are displayed in the UI.

### 🚀 Run Gradio:
```python
export disease_backend=your_api_token
python gradio.py
```

# 🧠 Models & Technologies Used

ClinicalBERT – medical text embeddings

BioGPT – enhanced medical language understanding

RAG (Retrieval-Augmented Generation) – enriches symptom context

FastAPI – REST API backend

Gradio – frontend interface



