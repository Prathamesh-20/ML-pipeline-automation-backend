"""
Dataset Upload Router
Handles CSV file uploads and dataset management.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
import uuid
from datetime import datetime

router = APIRouter()

# In-memory storage for datasets (in production, use a database)
datasets = {}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset file.
    Returns dataset ID and basic information.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    # Generate unique dataset ID
    dataset_id = str(uuid.uuid4())[:8]
    
    # Save file
    file_path = f"uploads/{dataset_id}_{file.filename}"
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Read and validate as DataFrame
        df = pd.read_csv(file_path)
        
        # Store dataset info
        datasets[dataset_id] = {
            "id": dataset_id,
            "filename": file.filename,
            "filepath": file_path,
            "uploaded_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        
        # Get preview data (first 10 rows)
        preview = df.head(10).to_dict(orient="records")
        
        return {
            "success": True,
            "message": "Dataset uploaded successfully!",
            "dataset_id": dataset_id,
            "info": {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "preview": preview
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/datasets")
async def list_datasets():
    """Get all uploaded datasets"""
    return {"datasets": list(datasets.values())}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset info by ID"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets[dataset_id]


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, rows: int = 20):
    """Get dataset preview with specified number of rows"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    filepath = datasets[dataset_id]["filepath"]
    df = pd.read_csv(filepath)
    
    return {
        "dataset_id": dataset_id,
        "total_rows": len(df),
        "preview_rows": min(rows, len(df)),
        "data": df.head(rows).to_dict(orient="records"),
        "columns": df.columns.tolist()
    }


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete file
    filepath = datasets[dataset_id]["filepath"]
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Remove from storage
    del datasets[dataset_id]
    
    return {"success": True, "message": "Dataset deleted successfully"}
