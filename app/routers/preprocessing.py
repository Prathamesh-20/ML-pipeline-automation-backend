"""
Data Preprocessing Router
Handles data cleaning and transformation operations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

router = APIRouter()

# Store preprocessed datasets
preprocessed_datasets = {}


class PreprocessingRequest(BaseModel):
    dataset_id: str
    steps: List[Dict[str, Any]]
    """
    Example steps:
    [
        {"action": "handle_missing", "method": "fill_mean", "columns": ["age", "salary"]},
        {"action": "encode_categorical", "method": "label_encoding", "columns": ["gender"]},
        {"action": "scale_features", "method": "standard_scaler", "columns": ["age", "salary"]}
    ]
    """


def handle_missing_values(df: pd.DataFrame, method: str, columns: List[str]) -> pd.DataFrame:
    """Handle missing values in specified columns."""
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == "fill_mean":
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
        elif method == "fill_median":
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
        elif method == "fill_mode":
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
        elif method == "drop_rows":
            df = df.dropna(subset=[col])
        elif method == "fill_zero":
            df[col] = df[col].fillna(0)
    
    return df


def encode_categorical(df: pd.DataFrame, method: str, columns: List[str]) -> tuple:
    """Encode categorical columns."""
    df = df.copy()
    encoders = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == "label_encoding":
            le = LabelEncoder()
            # Handle missing values before encoding
            df[col] = df[col].fillna("Missing")
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = {
                "type": "label",
                "classes": le.classes_.tolist()
            }
        elif method == "one_hot_encoding":
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            encoders[col] = {
                "type": "one_hot",
                "columns": dummies.columns.tolist()
            }
    
    return df, encoders


def scale_features(df: pd.DataFrame, method: str, columns: List[str]) -> tuple:
    """Scale numeric features."""
    df = df.copy()
    scaler_info = {}
    
    valid_columns = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    if not valid_columns:
        return df, scaler_info
    
    if method == "standard_scaler":
        scaler = StandardScaler()
        df[valid_columns] = scaler.fit_transform(df[valid_columns])
        scaler_info = {
            "type": "standard",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "columns": valid_columns
        }
    elif method == "min_max_scaler":
        scaler = MinMaxScaler()
        df[valid_columns] = scaler.fit_transform(df[valid_columns])
        scaler_info = {
            "type": "min_max",
            "min": scaler.data_min_.tolist(),
            "max": scaler.data_max_.tolist(),
            "columns": valid_columns
        }
    
    return df, scaler_info


@router.post("/preprocess")
async def preprocess_dataset(request: PreprocessingRequest):
    """
    Apply preprocessing steps to a dataset.
    Returns the preprocessed dataset info.
    """
    from app.routers.upload import datasets
    
    dataset_id = request.dataset_id
    
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    filepath = datasets[dataset_id]["filepath"]
    df = pd.read_csv(filepath)
    
    applied_steps = []
    transformation_log = []
    encoders = {}
    scaler_info = {}
    
    for step in request.steps:
        action = step.get("action")
        method = step.get("method")
        columns = step.get("columns", [])
        
        original_shape = df.shape
        
        try:
            if action == "handle_missing":
                df = handle_missing_values(df, method, columns)
                transformation_log.append({
                    "action": "handle_missing",
                    "method": method,
                    "columns": columns,
                    "rows_before": original_shape[0],
                    "rows_after": df.shape[0]
                })
                
            elif action == "encode_categorical":
                df, step_encoders = encode_categorical(df, method, columns)
                encoders.update(step_encoders)
                transformation_log.append({
                    "action": "encode_categorical",
                    "method": method,
                    "columns": columns,
                    "new_columns": list(set(df.columns) - set(columns)) if method == "one_hot_encoding" else columns
                })
                
            elif action == "scale_features":
                df, step_scaler_info = scale_features(df, method, columns)
                scaler_info.update(step_scaler_info)
                transformation_log.append({
                    "action": "scale_features",
                    "method": method,
                    "columns": columns
                })
            
            applied_steps.append(step)
            
        except Exception as e:
            transformation_log.append({
                "action": action,
                "error": str(e)
            })
    
    # Save preprocessed dataset
    preprocessed_id = f"{dataset_id}_preprocessed"
    preprocessed_path = f"uploads/{preprocessed_id}.csv"
    df.to_csv(preprocessed_path, index=False)
    
    # Store preprocessed info
    preprocessed_datasets[preprocessed_id] = {
        "id": preprocessed_id,
        "original_dataset_id": dataset_id,
        "filepath": preprocessed_path,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "encoders": encoders,
        "scaler_info": scaler_info,
        "applied_steps": applied_steps
    }
    
    return {
        "success": True,
        "message": "Preprocessing completed successfully!",
        "preprocessed_dataset_id": preprocessed_id,
        "transformation_log": transformation_log,
        "result": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "preview": df.head(10).to_dict(orient="records")
        },
        "encoders": encoders,
        "user_guidance": {
            "message": "Your data is now ready for training!",
            "next_step": "Select a target column and algorithm to train your model"
        }
    }


@router.get("/preprocess/{dataset_id}/auto")
async def auto_preprocess(dataset_id: str):
    """
    Automatically preprocess a dataset with recommended settings.
    """
    from app.routers.upload import datasets
    
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    filepath = datasets[dataset_id]["filepath"]
    df = pd.read_csv(filepath)
    
    # Auto-generate preprocessing steps
    auto_steps = []
    
    # Step 1: Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        numeric_missing = [col for col in missing_cols if df[col].dtype in ['int64', 'float64']]
        categorical_missing = [col for col in missing_cols if df[col].dtype == 'object']
        
        if numeric_missing:
            auto_steps.append({
                "action": "handle_missing",
                "method": "fill_median",
                "columns": numeric_missing
            })
        if categorical_missing:
            auto_steps.append({
                "action": "handle_missing",
                "method": "fill_mode",
                "columns": categorical_missing
            })
    
    # Step 2: Encode categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        auto_steps.append({
            "action": "encode_categorical",
            "method": "label_encoding",
            "columns": categorical_cols
        })
    
    # Step 3: Scale numeric (optional - only if high variance)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        ranges = df[numeric_cols].max() - df[numeric_cols].min()
        if ranges.max() / (ranges.min() + 0.001) > 10:
            auto_steps.append({
                "action": "scale_features",
                "method": "standard_scaler",
                "columns": numeric_cols
            })
    
    return {
        "dataset_id": dataset_id,
        "suggested_steps": auto_steps,
        "message": "These are the recommended preprocessing steps based on your data analysis",
        "user_action": "Review and click 'Apply' to preprocess, or customize the steps"
    }
