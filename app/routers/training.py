"""
Model Training Router
Handles ML model training and evaluation.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import uuid
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA

router = APIRouter()

# Store training jobs and trained models
training_jobs = {}
trained_models = {}


class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str  # classification, regression, clustering
    algorithm: str
    test_size: float = 0.2
    hyperparameters: Optional[Dict[str, Any]] = None
    use_cross_validation: bool = True


def get_model_instance(algorithm: str, hyperparameters: Dict = None):
    """Get sklearn model instance based on algorithm name."""
    params = hyperparameters or {}
    
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=42
        ),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=42
        ),
        "svm": SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True,
            random_state=42
        ),
        "logistic_regression": LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=42
        ),
        "linear_regression": LinearRegression(),
        "kmeans": KMeans(
            n_clusters=params.get("n_clusters", 3),
            random_state=42
        )
    }
    
    return models.get(algorithm)


def calculate_feature_importance(model, feature_names: List[str]) -> List[Dict]:
    """Extract feature importance from trained model."""
    importance_list = []
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            importance_list.append({
                "feature": name,
                "importance": round(float(importance), 4)
            })
        importance_list.sort(key=lambda x: x["importance"], reverse=True)
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if len(coefs.shape) > 1:
            coefs = coefs[0]
        for name, coef in zip(feature_names, coefs):
            importance_list.append({
                "feature": name,
                "importance": round(float(abs(coef)), 4)
            })
        importance_list.sort(key=lambda x: x["importance"], reverse=True)
    
    return importance_list


@router.post("/train")
async def train_model(request: TrainingRequest):
    """
    Train a machine learning model.
    Returns training results and model metrics.
    """
    from app.routers.upload import datasets
    from app.routers.preprocessing import preprocessed_datasets
    
    dataset_id = request.dataset_id
    
    # Check both original and preprocessed datasets
    if dataset_id in datasets:
        filepath = datasets[dataset_id]["filepath"]
    elif dataset_id in preprocessed_datasets:
        filepath = preprocessed_datasets[dataset_id]["filepath"]
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = pd.read_csv(filepath)
    
    # Validate target column
    if request.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
    
    # Prepare features and target
    X = df.drop(columns=[request.target_column])
    y = df[request.target_column]
    feature_names = X.columns.tolist()
    
    # Track preprocessing steps applied
    preprocessing_applied = []
    
    # Auto-preprocess: Handle missing values
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown')
        preprocessing_applied.append({
            "step": "Missing Value Handling",
            "description": f"Filled {missing_before} missing values with median (numeric) or mode (categorical)",
            "columns_affected": X.columns[X.isnull().any()].tolist() if X.isnull().any().any() else "All columns processed"
        })
    
    # Auto-preprocess: Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        preprocessing_applied.append({
            "step": "Categorical Encoding",
            "description": f"Applied Label Encoding to {len(categorical_cols)} text columns",
            "columns_affected": categorical_cols
        })
    
    # Encode target if it's categorical (for classification)
    target_encoder = None
    if y.dtype == 'object' or request.task_type == 'classification':
        target_encoder = LabelEncoder()
        original_classes = y.unique().tolist()
        y = target_encoder.fit_transform(y.astype(str))
        preprocessing_applied.append({
            "step": "Target Encoding",
            "description": f"Encoded target column '{request.target_column}' ({len(original_classes)} classes)",
            "columns_affected": [request.target_column]
        })
    
    # Get model instance
    model = get_model_instance(request.algorithm, request.hyperparameters)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    results = {
        "job_id": job_id,
        "dataset_id": dataset_id,
        "algorithm": request.algorithm,
        "task_type": request.task_type,
        "target_column": request.target_column,
        "started_at": datetime.now().isoformat(),
        "preprocessing_applied": preprocessing_applied
    }
    
    try:
        if request.task_type == "clustering":
            # Clustering doesn't need train/test split
            model.fit(X)
            labels = model.predict(X)
            
            # Metrics
            silhouette = silhouette_score(X, labels)
            
            # PCA for visualization
            pca_data = None
            if X.shape[1] > 1:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X)
                centroids_2d = pca.transform(model.cluster_centers_)
                pca_data = {
                    "points": [{"x": float(p[0]), "y": float(p[1]), "cluster": int(l)} 
                              for p, l in zip(pca_result, labels)],
                    "centroids": [{"x": float(c[0]), "y": float(c[1])} 
                                  for c in centroids_2d]
                }
            
            results.update({
                "success": True,
                "metrics": {
                    "silhouette_score": round(silhouette, 4),
                    "n_clusters": len(set(labels)),
                    "cluster_sizes": {str(i): int((labels == i).sum()) for i in set(labels)}
                },
                "visualization": pca_data,
                "interpretation": {
                    "silhouette_score": "Measures cluster quality (-1 to 1). Higher is better. "
                                       f"Your score of {silhouette:.2f} indicates "
                                       f"{'well-separated' if silhouette > 0.5 else 'moderate' if silhouette > 0.25 else 'overlapping'} clusters."
                }
            })
            
        else:
            # Classification or Regression - use train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=request.test_size, random_state=42
            )
            
            # Cross-validation
            cv_scores = None
            if request.use_cross_validation:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Train model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Feature importance
            feature_importance = calculate_feature_importance(model, feature_names)
            
            if request.task_type == "classification":
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, predictions)
                unique_labels = sorted(np.unique(y_test))
                
                results.update({
                    "success": True,
                    "metrics": {
                        "accuracy": round(accuracy, 4),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1_score": round(f1, 4),
                        "cv_mean": round(cv_scores.mean(), 4) if cv_scores is not None else None,
                        "cv_std": round(cv_scores.std(), 4) if cv_scores is not None else None
                    },
                    "confusion_matrix": {
                        "matrix": cm.tolist(),
                        "labels": [str(l) for l in unique_labels]
                    },
                    "feature_importance": feature_importance[:10],  # Top 10
                    "interpretation": {
                        "accuracy": f"Your model correctly predicts {accuracy*100:.1f}% of cases.",
                        "precision": f"When the model predicts positive, it's correct {precision*100:.1f}% of the time.",
                        "recall": f"The model finds {recall*100:.1f}% of all positive cases.",
                        "f1_score": f"F1 combines precision and recall. Score: {f1:.2f} (higher is better).",
                        "overall": "Good!" if accuracy > 0.8 else "Acceptable" if accuracy > 0.6 else "Needs improvement"
                    }
                })
                
            else:  # Regression
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                rmse = np.sqrt(mse)
                
                results.update({
                    "success": True,
                    "metrics": {
                        "mse": round(mse, 4),
                        "rmse": round(rmse, 4),
                        "mae": round(mae, 4),
                        "r2_score": round(r2, 4),
                        "cv_mean": round(cv_scores.mean(), 4) if cv_scores is not None else None,
                        "cv_std": round(cv_scores.std(), 4) if cv_scores is not None else None
                    },
                    "feature_importance": feature_importance[:10],
                    "predictions_sample": {
                        "actual": y_test.head(10).tolist(),
                        "predicted": predictions[:10].tolist()
                    },
                    "interpretation": {
                        "r2_score": f"R² of {r2:.2f} means the model explains {r2*100:.1f}% of the variance.",
                        "mae": f"On average, predictions are off by {mae:.2f} units.",
                        "rmse": f"Root Mean Square Error: {rmse:.2f} (lower is better).",
                        "overall": "Excellent!" if r2 > 0.9 else "Good" if r2 > 0.7 else "Moderate" if r2 > 0.5 else "Needs improvement"
                    }
                })
        
        # Save model
        model_path = f"trained_models/{job_id}_model.pkl"
        joblib.dump(model, model_path)
        
        trained_models[job_id] = {
            "job_id": job_id,
            "model_path": model_path,
            "algorithm": request.algorithm,
            "task_type": request.task_type,
            "feature_names": feature_names,
            "target_column": request.target_column,
            "created_at": datetime.now().isoformat()
        }
        
        results["model_saved"] = True
        results["model_id"] = job_id
        results["completed_at"] = datetime.now().isoformat()
        
        # Add user guidance
        results["user_guidance"] = {
            "message": "Training complete! Your model has been saved.",
            "next_steps": [
                "Download the trained model for use in other applications",
                "Make predictions on new data using the /predict endpoint",
                "Try a different algorithm to compare results"
            ]
        }
        
    except Exception as e:
        results.update({
            "success": False,
            "error": str(e),
            "user_guidance": {
                "message": f"Training failed: {str(e)}",
                "suggestions": [
                    "Make sure your data is properly preprocessed",
                    "Check if the target column is appropriate for this task type",
                    "Try a different algorithm"
                ]
            }
        })
    
    training_jobs[job_id] = results
    return results


@router.get("/train/{job_id}")
async def get_training_result(job_id: str):
    """Get training job result by ID."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_jobs[job_id]


@router.get("/models")
async def list_trained_models():
    """List all trained models."""
    return {"models": list(trained_models.values())}


@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download trained model as a .pkl file."""
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models[model_id]
    model_path = model_info["model_path"]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")
    
    return FileResponse(
        path=model_path,
        filename=f"{model_info['algorithm']}_{model_id}.pkl",
        media_type="application/octet-stream"
    )


class PredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]


@router.post("/predict")
async def make_prediction(request: PredictionRequest):
    """Make predictions using a trained model."""
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models[request.model_id]
    model_path = model_info["model_path"]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Load model
    model = joblib.load(model_path)
    
    # Convert input data to DataFrame
    df = pd.DataFrame(request.data)
    
    # Ensure columns match training data
    expected_features = model_info.get("feature_names", [])
    if expected_features:
        # Check for missing columns
        missing = set(expected_features) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {list(missing)}"
            )
        # Reorder to match training
        df = df[expected_features]
    
    # Make predictions
    try:
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df).tolist()
        
        return {
            "success": True,
            "model_id": request.model_id,
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/algorithms")
async def list_algorithms():
    """List all available algorithms with descriptions."""
    return {
        "classification": [
            {
                "id": "random_forest",
                "name": "Random Forest",
                "description": "Ensemble of decision trees - great for most classification tasks",
                "hyperparameters": [
                    {"name": "n_estimators", "type": "int", "default": 100, "range": [10, 500]},
                    {"name": "max_depth", "type": "int", "default": None, "range": [1, 50]}
                ]
            },
            {
                "id": "svm",
                "name": "Support Vector Machine",
                "description": "Finds optimal decision boundary - effective for smaller datasets",
                "hyperparameters": [
                    {"name": "C", "type": "float", "default": 1.0, "range": [0.01, 10]},
                    {"name": "kernel", "type": "choice", "default": "rbf", "options": ["linear", "rbf", "poly"]}
                ]
            },
            {
                "id": "logistic_regression",
                "name": "Logistic Regression",
                "description": "Simple and interpretable - good baseline for binary classification",
                "hyperparameters": [
                    {"name": "C", "type": "float", "default": 1.0, "range": [0.01, 10]}
                ]
            }
        ],
        "regression": [
            {
                "id": "random_forest_regressor",
                "name": "Random Forest Regressor",
                "description": "Ensemble method for predicting continuous values",
                "hyperparameters": [
                    {"name": "n_estimators", "type": "int", "default": 100, "range": [10, 500]},
                    {"name": "max_depth", "type": "int", "default": None, "range": [1, 50]}
                ]
            },
            {
                "id": "linear_regression",
                "name": "Linear Regression",
                "description": "Simple linear model - very fast and interpretable",
                "hyperparameters": []
            }
        ],
        "clustering": [
            {
                "id": "kmeans",
                "name": "K-Means Clustering",
                "description": "Groups data into K clusters based on similarity",
                "hyperparameters": [
                    {"name": "n_clusters", "type": "int", "default": 3, "range": [2, 20]}
                ]
            }
        ]
    }
