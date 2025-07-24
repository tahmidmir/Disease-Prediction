from app.schemas import DiseaseOutput

def clean_output(raw_text: str) -> DiseaseOutput:
    diseases = [d.strip() for d in raw_text.split(",") if d.strip()]
    return DiseaseOutput(predictions=diseases)
