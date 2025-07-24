import torch
from app.models.load_models import (
    clinicalbert_model, clinicalbert_tokenizer,
    biogpt_model, biogpt_tokenizer,
    load_model_and_artifacts, load_rag_knowledge,
    device
)
from app.utils.rag import rag_enhance
from app.schemas import DiseaseOutput

model, label_encoder = load_model_and_artifacts()
rag_index, rag_texts = load_rag_knowledge()

def predict_pipeline(symptoms: str) -> DiseaseOutput:
    enhanced, rag_disease, rag_treatment = rag_enhance(symptoms, rag_index, rag_texts)

    if label_encoder:
        inputs = clinicalbert_tokenizer(enhanced, return_tensors="pt", max_length=128, truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = clinicalbert_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            logits = model(emb)
            pred = torch.argmax(logits, dim=1).item()
            try:
                disease = label_encoder.inverse_transform([pred])[0]
            except:
                disease = rag_disease
    else:
        disease = rag_disease

    treatment = rag_treatment or "Consult a doctor"
    return DiseaseOutput(
        enhanced_symptoms=enhanced,
        predicted_disease=disease,
        treatment=treatment
    )
