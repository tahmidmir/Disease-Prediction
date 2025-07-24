import re
import torch
import faiss
import pickle
from transformers import BioGptTokenizer, BioGptForCausalLM
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BioGPT from Hugging Face Hub
biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT").to(device)
biogpt_model.eval()

# Load BioBERT sentence encoder from Hugging Face Hub
rag_encoder = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1", device=device)

# Load RAG knowledge base (FAISS index + texts)
def load_rag_knowledge(save_folder="saved_models/static"):
    rag_index_path = f"{save_folder}/rag_index.faiss"
    rag_texts_path = f"{save_folder}/rag_texts.pkl"

    try:
        with open(rag_texts_path, "rb") as f:
            rag_texts = pickle.load(f)
        rag_index = faiss.read_index(rag_index_path)
        print(f"✅ Loaded RAG with {len(rag_texts)} chunks.")
    except Exception as e:
        print(f"⚠️ Could not load RAG: {e}")
        rag_index = faiss.IndexFlatL2(768)
        rag_texts = []

    return rag_index, rag_texts

# Enhance user symptoms with BioGPT + RAG
def rag_enhance(symptoms: str, index, pdf_texts: list, top_k: int = 3):
    if not pdf_texts or index.ntotal == 0:
        return symptoms, "Unknown (RAG unavailable)", "Consult a healthcare provider"

    symptom_embedding = rag_encoder.encode([symptoms], convert_to_tensor=True, device=device).cpu().numpy()
    distances, indices = index.search(symptom_embedding, top_k)

    retrieved_chunks = [pdf_texts[i] for i in indices[0] if i < len(pdf_texts)]

    context = f"Symptoms: {symptoms}\nRelevant Information: {' '.join(retrieved_chunks)}"
    prompt = (
        f"Based on the following information:\n{context}\n"
        "Provide a concise response with:\n"
        "1. Enhanced Symptoms:\n"
        "2. Disease:\n"
        "3. Treatment:"
    )

    input_ids = biogpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = biogpt_model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.33
        )

    generated_text = biogpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract structured output with regex
    enhanced_symptoms = symptoms
    disease = "Unknown"
    treatment = "Consult a healthcare provider"

    sym_match = re.search(r"Enhanced Symptoms:\s*(.+?)(?=\n|$|Disease:)", generated_text, re.DOTALL)
    dis_match = re.search(r"Disease:\s*(.+?)(?=\n|$|Treatment:)", generated_text, re.DOTALL)
    treat_match = re.search(r"Treatment:\s*(.+?)(?=\n|$)", generated_text, re.DOTALL)

    if sym_match:
        enhanced_symptoms = sym_match.group(1).strip()
    if dis_match:
        disease = dis_match.group(1).strip()
    if treat_match:
        treatment = treat_match.group(1).strip()

    return enhanced_symptoms, disease, treatment
