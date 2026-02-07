"""
Data Analysis Router
Provides smart data analysis and recommendations for users.
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from typing import Dict, Any, List

router = APIRouter()


def detect_task_type(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    Detect recommended ML task type based on data characteristics.
    """
    if target_column and target_column in df.columns:
        target = df[target_column]
        unique_ratio = len(target.unique()) / len(target)
        
        if target.dtype == 'object' or unique_ratio < 0.05:
            return {
                "recommended_task": "classification",
                "confidence": 0.9,
                "reason": f"Target column '{target_column}' has {len(target.unique())} unique categories"
            }
        elif unique_ratio > 0.3:
            return {
                "recommended_task": "regression",
                "confidence": 0.85,
                "reason": f"Target column '{target_column}' appears to be continuous with high variance"
            }
    
    return {
        "recommended_task": "clustering",
        "confidence": 0.7,
        "reason": "No clear target column - consider unsupervised learning"
    }


def calculate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality score and identify issues.
    """
    issues = []
    total_score = 100
    
    # Check missing values
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct > 0:
        issues.append({
            "type": "missing_values",
            "severity": "high" if missing_pct > 10 else "medium" if missing_pct > 5 else "low",
            "message": f"{missing_pct:.1f}% of data is missing",
            "columns": df.columns[df.isnull().any()].tolist(),
            "suggestion": "Consider filling with mean/median or dropping rows"
        })
        total_score -= min(30, missing_pct * 2)
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append({
            "type": "duplicates",
            "severity": "medium",
            "message": f"{duplicate_rows} duplicate rows found",
            "suggestion": "Consider removing duplicate entries"
        })
        total_score -= min(10, (duplicate_rows / len(df)) * 100)
    
    # Check for high cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = len(df[col].unique()) / len(df)
        if unique_ratio > 0.5:
            issues.append({
                "type": "high_cardinality",
                "severity": "medium",
                "message": f"Column '{col}' has very high cardinality ({len(df[col].unique())} unique values)",
                "suggestion": "Consider removing or encoding this column differently"
            })
            total_score -= 5
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append({
                "type": "constant_column",
                "severity": "low",
                "message": f"Column '{col}' has only one unique value",
                "suggestion": "This column provides no information - consider removing"
            })
            total_score -= 5
    
    return {
        "score": max(0, int(total_score)),
        "grade": "A" if total_score >= 90 else "B" if total_score >= 80 else "C" if total_score >= 70 else "D" if total_score >= 60 else "F",
        "issues": issues,
        "is_ready": total_score >= 70
    }


def get_column_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Analyze each column and provide insights.
    """
    columns_analysis = []
    
    for col in df.columns:
        col_data = df[col]
        analysis = {
            "name": col,
            "dtype": str(col_data.dtype),
            "missing": int(col_data.isnull().sum()),
            "missing_pct": round(col_data.isnull().sum() / len(col_data) * 100, 2),
            "unique": int(col_data.nunique()),
            "unique_pct": round(col_data.nunique() / len(col_data) * 100, 2)
        }
        
        if col_data.dtype in ['int64', 'float64']:
            analysis.update({
                "is_numeric": True,
                "mean": round(col_data.mean(), 4) if not col_data.isnull().all() else None,
                "std": round(col_data.std(), 4) if not col_data.isnull().all() else None,
                "min": float(col_data.min()) if not col_data.isnull().all() else None,
                "max": float(col_data.max()) if not col_data.isnull().all() else None,
                "suggested_role": "feature" if col_data.nunique() > 2 else "target_candidate"
            })
        else:
            analysis.update({
                "is_numeric": False,
                "top_values": col_data.value_counts().head(5).to_dict(),
                "suggested_role": "target_candidate" if col_data.nunique() < 20 else "categorical_feature"
            })
        
        columns_analysis.append(analysis)
    
    return columns_analysis


def get_preprocessing_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate preprocessing recommendations.
    """
    recommendations = []
    
    # Missing values recommendation
    if df.isnull().any().any():
        missing_cols = df.columns[df.isnull().any()].tolist()
        recommendations.append({
            "step": 1,
            "action": "handle_missing_values",
            "title": "Handle Missing Values",
            "description": f"Found missing values in {len(missing_cols)} columns",
            "columns": missing_cols,
            "options": ["fill_mean", "fill_median", "fill_mode", "drop_rows"],
            "recommended": "fill_median",
            "importance": "required"
        })
    
    # Categorical encoding recommendation
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        recommendations.append({
            "step": 2,
            "action": "encode_categorical",
            "title": "Encode Categorical Variables",
            "description": f"Found {len(categorical_cols)} text/categorical columns that need encoding",
            "columns": categorical_cols,
            "options": ["label_encoding", "one_hot_encoding"],
            "recommended": "label_encoding",
            "importance": "required"
        })
    
    # Scaling recommendation
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        # Check for different scales
        ranges = df[numeric_cols].max() - df[numeric_cols].min()
        if ranges.max() / (ranges.min() + 0.001) > 10:
            recommendations.append({
                "step": 3,
                "action": "scale_features",
                "title": "Scale Numeric Features",
                "description": "Numeric features have different scales - normalization recommended",
                "columns": numeric_cols,
                "options": ["standard_scaler", "min_max_scaler", "none"],
                "recommended": "standard_scaler",
                "importance": "recommended"
            })
    
    return recommendations


def get_algorithm_recommendations(task_type: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Recommend algorithms based on task type and data characteristics.
    """
    n_samples = len(df)
    n_features = len(df.columns) - 1  # Assuming one target column
    
    algorithms = []
    
    if task_type == "classification":
        algorithms = [
            {
                "name": "Random Forest",
                "id": "random_forest",
                "description": "Ensemble of decision trees - handles many features well",
                "pros": ["Robust to outliers", "Feature importance built-in", "Good accuracy"],
                "cons": ["Can be slow on large datasets", "Less interpretable"],
                "recommended": n_features > 5,
                "beginner_friendly": True
            },
            {
                "name": "Support Vector Machine",
                "id": "svm",
                "description": "Finds optimal boundary between classes",
                "pros": ["Effective in high dimensions", "Memory efficient"],
                "cons": ["Slower on large datasets", "Needs feature scaling"],
                "recommended": n_samples < 10000,
                "beginner_friendly": True
            },
            {
                "name": "Logistic Regression",
                "id": "logistic_regression",
                "description": "Simple linear classifier - highly interpretable",
                "pros": ["Fast", "Interpretable", "Works well for binary classification"],
                "cons": ["Limited to linear relationships"],
                "recommended": n_features < 20,
                "beginner_friendly": True
            }
        ]
    elif task_type == "regression":
        algorithms = [
            {
                "name": "Random Forest Regressor",
                "id": "random_forest_regressor",
                "description": "Ensemble method for continuous predictions",
                "pros": ["Handles non-linear relationships", "Robust"],
                "cons": ["Can overfit on noisy data"],
                "recommended": True,
                "beginner_friendly": True
            },
            {
                "name": "Linear Regression",
                "id": "linear_regression",
                "description": "Simple linear model - fastest and most interpretable",
                "pros": ["Very fast", "Highly interpretable", "Good baseline"],
                "cons": ["Assumes linear relationships"],
                "recommended": n_features < 10,
                "beginner_friendly": True
            }
        ]
    else:  # clustering
        algorithms = [
            {
                "name": "K-Means",
                "id": "kmeans",
                "description": "Groups data into K clusters based on similarity",
                "pros": ["Fast", "Easy to understand", "Scales well"],
                "cons": ["Must specify K in advance", "Sensitive to outliers"],
                "recommended": True,
                "beginner_friendly": True
            }
        ]
    
    return algorithms


@router.get("/analyze/{dataset_id}")
async def analyze_dataset(dataset_id: str, target_column: str = None):
    """
    Perform comprehensive analysis on a dataset.
    Returns quality score, recommendations, and insights.
    """
    from app.routers.upload import datasets
    
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    filepath = datasets[dataset_id]["filepath"]
    df = pd.read_csv(filepath)
    
    # Perform analysis
    data_quality = calculate_data_quality(df)
    columns_analysis = get_column_analysis(df)
    task_detection = detect_task_type(df, target_column)
    preprocessing_recommendations = get_preprocessing_recommendations(df)
    algorithm_recommendations = get_algorithm_recommendations(
        task_detection["recommended_task"], df
    )
    
    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = {}
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        correlation = corr_matrix.to_dict()
    
    return {
        "dataset_id": dataset_id,
        "summary": {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['int64', 'float64']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        },
        "data_quality": data_quality,
        "columns": columns_analysis,
        "task_recommendation": task_detection,
        "preprocessing_steps": preprocessing_recommendations,
        "algorithm_recommendations": algorithm_recommendations,
        "correlation": correlation,
        "user_guidance": {
            "next_step": "preprocessing" if not data_quality["is_ready"] else "training",
            "message": "Your data needs preprocessing before training" if not data_quality["is_ready"] 
                      else "Your data is ready! Select an algorithm and start training"
        }
    }
