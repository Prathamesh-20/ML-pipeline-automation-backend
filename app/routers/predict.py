"""
Prediction Router
Handles making predictions with trained models.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib

router = APIRouter()


class PredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]  # List of records to predict


@router.post("/predict")
async def make_prediction(request: PredictionRequest):
    """
    Make predictions using a trained model.
    """
    from app.routers.training import trained_models
    
    model_id = request.model_id
    
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models[model_id]
    
    try:
        # Load model
        model = joblib.load(model_info["model_path"])
        
        # Prepare input data
        df = pd.DataFrame(request.data)
        
        # Validate features
        expected_features = model_info["feature_names"]
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing features: {missing_features}"
            )
        
        # Reorder columns to match training
        df = df[expected_features]
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available (classification)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)
            if hasattr(model, 'classes_'):
                probabilities = [
                    {str(cls): round(float(p), 4) for cls, p in zip(model.classes_, prob)}
                    for prob in proba
                ]
        
        return {
            "success": True,
            "model_id": model_id,
            "algorithm": model_info["algorithm"],
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "num_predictions": len(predictions),
            "user_guidance": {
                "message": f"Successfully generated {len(predictions)} predictions!",
                "interpretation": "Each prediction corresponds to a row in your input data"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/{model_id}/file")
async def predict_from_file(model_id: str, file: UploadFile = File(...)):
    """
    Make predictions on a CSV file using a trained model.
    """
    from app.routers.training import trained_models
    
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    model_info = trained_models[model_id]
    
    try:
        # Read file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Load model
        model = joblib.load(model_info["model_path"])
        
        # Validate and prepare features
        expected_features = model_info["feature_names"]
        missing_features = set(expected_features) - set(df.columns)
        if missing_features:
            return {
                "success": False,
                "error": f"Missing features in file: {list(missing_features)}",
                "expected_features": expected_features,
                "found_features": df.columns.tolist()
            }
        
        # Select and reorder features
        X = df[expected_features]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if hasattr(model, 'classes_'):
                for i, cls in enumerate(model.classes_):
                    df[f'probability_{cls}'] = proba[:, i]
        
        return {
            "success": True,
            "model_id": model_id,
            "total_predictions": len(predictions),
            "preview": df.head(20).to_dict(orient="records"),
            "prediction_summary": {
                "unique_predictions": len(set(predictions)),
                "value_counts": pd.Series(predictions).value_counts().to_dict()
            },
            "user_guidance": {
                "message": f"Generated {len(predictions)} predictions from your file!",
                "columns_added": ["prediction"] + 
                               [f"probability_{cls}" for cls in getattr(model, 'classes_', [])]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/predict/{model_id}/sample-input")
async def get_sample_input(model_id: str):
    """
    Get a sample input format for making predictions with a model.
    """
    from app.routers.training import trained_models
    
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models[model_id]
    
    # Create sample record with placeholder values
    sample_record = {feature: "<value>" for feature in model_info["feature_names"]}
    
    return {
        "model_id": model_id,
        "required_features": model_info["feature_names"],
        "sample_request": {
            "model_id": model_id,
            "data": [sample_record]
        },
        "user_guidance": {
            "message": "Use this format to make predictions",
            "steps": [
                "Replace <value> with actual values for each feature",
                "Add multiple objects to the 'data' array for batch predictions",
                "Send a POST request to /api/predict"
            ]
        }
    }
