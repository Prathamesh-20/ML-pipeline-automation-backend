# ML Pipeline Automation - Backend

A FastAPI-based backend for no-code machine learning pipeline automation, providing REST APIs for data upload, analysis, preprocessing, and model training.

## 🚀 Features

- **Data Upload**: CSV file upload with automatic data type detection
- **Smart Analysis**: Data quality scoring, column analysis, task type detection
- **Auto-Preprocessing**: Automatic label encoding and missing value handling
- **Model Training**: Support for classification, regression, and clustering
- **Model Export**: Download trained models as `.pkl` files
- **Predictions API**: Make predictions using trained models

## 📋 Supported Algorithms

| Task | Algorithms |
|------|------------|
| Classification | Random Forest, SVM, Logistic Regression |
| Regression | Random Forest Regressor, Linear Regression |
| Clustering | K-Means |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Prathamesh-20/ML-pipeline-automation-backend.git
cd ML-pipeline-automation-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the Server

```bash
python run.py
```

Server runs at: `http://localhost:8000`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload CSV dataset |
| GET | `/api/datasets` | List all datasets |
| GET | `/api/analyze/{dataset_id}` | Analyze dataset |
| POST | `/api/train` | Train a model |
| GET | `/api/models/{model_id}/download` | Download trained model |
| POST | `/api/predict` | Make predictions |
| GET | `/api/algorithms` | List available algorithms |

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py          # FastAPI application entry
│   └── routers/
│       ├── upload.py    # File upload endpoints
│       ├── analysis.py  # Data analysis endpoints
│       ├── preprocessing.py  # Preprocessing endpoints
│       └── training.py  # Model training endpoints
├── requirements.txt
└── run.py
```

## 🔗 Related

- [Frontend Repository](https://github.com/Prathamesh-20/ML-pipeline-automation-frontend)

## 📄 License

MIT License
