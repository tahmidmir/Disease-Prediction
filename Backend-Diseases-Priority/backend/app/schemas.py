from pydantic import BaseModel

class SymptomInput(BaseModel):
    symptoms: str

class DiseaseOutput(BaseModel):
    enhanced_symptoms: str
    predicted_disease: str
    treatment: str
