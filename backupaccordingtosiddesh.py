import os
import logging
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
import PyPDF2
import io

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Global variables to store model components
pipeline = None
model = None
kmeans = None
cluster_centers = None
cluster_to_kpi = None
model_loaded = False

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

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

@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Request received")
        
        logging.info(f"Request files: {request.files}")
        logging.info(f"Request form: {request.form}")
        
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logging.warning("No file selected for uploading")
            return jsonify({"error": "No file selected for uploading"}), 400
        
        logging.info(f"Received file: {file.filename}")
        
        # Read the uploaded file
        contents = file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf(contents)
        
        logging.info(f"Extracted text (first 100 chars): {text[:100]}")
        
        # Perform prediction using the loaded model components
        predicted_kpis = categorize_guideline(text, model, kmeans, cluster_centers, cluster_to_kpi)

        logging.info("Request processed successfully")
        # Return the result
        return jsonify({"related_kpis": predicted_kpis})
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

def initialize_app():
    try:
        load_model()
        app.logger.info("Application initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize application: {str(e)}")

if __name__ == "__main__":
    app.logger.info("Starting the Flask application...")
    initialize_app()
    try:
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        app.logger.error(f"Failed to start the Flask application: {str(e)}")