import os
import pickle
import torch
import faiss
from transformers import AutoTokenizer, AutoModel, BioGptTokenizer, BioGptForCausalLM
from sentence_transformers import SentenceTransformer
from app.models.classifier import ClinicalBERTDiseaseClassifier
from transformers import BertTokenizer, BertModel
import warnings

device = torch.device("cpu")
SAVE_FOLDER = "saved_models/static"


try:
    clinicalbert_tokenizer = BertTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        cache_dir="/code/.hf_cache"
    )
    clinicalbert_model = BertModel.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        cache_dir="/code/.hf_cache"
    ).to(device)
    clinicalbert_model.eval()
except Exception as e:
    print(f"❌ Failed to load ClinicalBERT: {str(e)}")
    raise


try:
    biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
    biogpt_model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT").to(device)
    biogpt_model.eval()
except Exception as e:
    print(f"❌ Failed to load BioGPT: {str(e)}")
    raise


try:
    rag_encoder = SentenceTransformer(
        "dmis-lab/biobert-base-cased-v1.1",
        device=device,
        cache_folder="/code/.hf_cache"
    )
    print("✅ BioBERT loaded successfully")
except Exception as e:
    print(f"❌ Failed to load BioBERT: {str(e)}")
    raise

def load_model_and_artifacts():

    try:

        os.makedirs(SAVE_FOLDER, exist_ok=True)
        
        label_encoder = None
        num_classes = 1

        label_path = os.path.join(SAVE_FOLDER, "label_encoder.pkl")
        if os.path.exists(label_path):
            with open(label_path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    label_encoder = pickle.load(f)
            num_classes = len(label_encoder.classes_)
            print(f"✅ Loaded label encoder with {num_classes} classes")
        else:
            print("⚠️ Label encoder not found")

        model = ClinicalBERTDiseaseClassifier(input_dim=768, num_classes=num_classes).to(device)
        model_path = os.path.join(SAVE_FOLDER, "best_model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded best model")
        else:
            print("⚠️ No trained model found")

        model.eval()
        return model, label_encoder

    except Exception as e:
        print(f"🔥 Error in load_model_and_artifacts: {str(e)}")
        raise

def load_rag_knowledge():

    try:
        rag_texts_path = os.path.join(SAVE_FOLDER, "rag_texts.pkl")
        rag_index_path = os.path.join(SAVE_FOLDER, "rag_index.faiss")

        if not all([os.path.exists(rag_texts_path), os.path.exists(rag_index_path)]):
            print("⚠️ Some RAG files are missing")
            return faiss.IndexFlatL2(768), []

        with open(rag_texts_path, "rb") as f:
            rag_texts = pickle.load(f)

        try:
            rag_index = faiss.read_index(rag_index_path)
        except RuntimeError:
            print("⚠️ FAISS index version mismatch, creating new index")
            return faiss.IndexFlatL2(768), rag_texts

        print(f"✅ Loaded RAG with {len(rag_texts)} chunks")
        return rag_index, rag_texts

    except Exception as e:
        print(f"🔥 Error in load_rag_knowledge: {str(e)}")
        return faiss.IndexFlatL2(768), []