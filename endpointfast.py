import os
import logging
import json
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import PyPDF2
import io
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.DEBUG)

# Global variables to store model components
pipeline = None
model = None
kmeans = None
cluster_centers = None
cluster_to_kpi = None
model_loaded = False

def load_model():
    global pipeline, model, kmeans, cluster_centers, cluster_to_kpi, model_loaded
    
    if not model_loaded:
        try:
            # Load the model pipeline
            model_path = "model_pipeline.pkl"  # Adjust this path as needed
            pipeline = joblib.load(model_path)

            # Extract components from the pipeline
            kmeans = pipeline['kmeans']
            cluster_centers = pipeline['cluster_centers']
            cluster_to_kpi = pipeline['cluster_to_kpi']
            model = pipeline['sbert_model']
            
            model_loaded = True
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        load_model()
        logging.info("Application initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize application: {str(e)}")
        raise
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

def extract_text_from_pdf(pdf_data):
    pdf_file = io.BytesIO(pdf_data)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def categorize_guideline(guideline, model, kmeans, cluster_centers, cluster_to_kpi):
    embedding = model.encode([guideline])[0]
    distances = np.linalg.norm(cluster_centers - embedding, axis=1)
    
    # Convert distances to relevance scores (higher distance -> lower relevance)
    relevance_scores = 1 / (1 + distances)
    relevance_scores = relevance_scores / relevance_scores.sum() * 100
    
    # Map the scores to KPIs and convert to native Python types
    kpi_relevance = {cluster_to_kpi[i]: float(relevance_scores[i]) for i in range(len(relevance_scores))}
    return kpi_relevance

@app.get("/")
async def home():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logging.info("Request received")
        
        if not file:
            raise HTTPException(status_code=400, detail="No file part in the request")
        
        logging.info(f"Received file: {file.filename}")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(contents)
        
        logging.info(f"Extracted text (first 100 chars): {text[:100]}")
        
        # Perform prediction using the loaded model components
        predicted_kpis = categorize_guideline(text, model, kmeans, cluster_centers, cluster_to_kpi)

        logging.info("Request processed successfully")
        # Return the result
        return JSONResponse(content={"related_kpis": predicted_kpis})
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)