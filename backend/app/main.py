from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.schemas import SymptomInput, DiseaseOutput
from app.pipeline import predict_pipeline
import os

os.environ["HF_HOME"] = "/code/.hf_cache"
# static_path = "app/frontend/build/static"

app = FastAPI()

# Serve static files
# app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

# # Serve React build files
# @app.get("/{full_path:path}")
# async def serve_react_app(full_path: str):
#     static_path = Path("app/frontend/static")
#     file_path = Path(f"app/frontend/{full_path}")
    
#     if file_path.exists() and file_path.is_file():
#         return FileResponse(file_path)
    
#     if (static_path / full_path).exists():
#         return FileResponse(static_path / full_path)
    
#     return FileResponse("app/frontend/index.html")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=DiseaseOutput)
def predict(input_data: SymptomInput):
    try:
        return predict_pipeline(input_data.symptoms)
    except Exception as e:
        return {"error": str(e)}