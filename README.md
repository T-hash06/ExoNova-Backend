# Exoplanet Detection Backend

FastAPI backend for exoplanet detection prediction using machine learning models.

## Requirements

- Python 3.10+
- UV package manager

## Installation

Install dependencies using UV:

```bash
uv sync
```

This will create a virtual environment and install all required dependencies including FastAPI, Uvicorn, and Pydantic.

## Running the Server

### Development Mode

Activate the virtual environment and run the server with auto-reload:

```bash
. .venv/bin/activate.fish
python src/main.py
```

The server will start on `http://localhost:8000` with auto-reload enabled for development.

### Production Mode

Run with Uvicorn directly:

```bash
. .venv/bin/activate.fish
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Mock Predictions

**IMPORTANT**: This implementation uses mock prediction functions that return realistic dummy data. No actual machine learning model inference is performed.

### Mock Function Locations

All mock prediction logic is located in:
```
src/ml/predictor.py
```

The mock functions include:
- `predict_tabular_mock()` - Tabular prediction with 21 parameters
- `predict_live_preview_mock()` - Live preview prediction with 6 parameters
- `_generate_mock_feature_weights()` - Helper for generating realistic feature weights

### Mock Behavior

- **Tabular predictions**: Returns random probability [0.0-1.0], confidence [0.7-0.95], and 15-20 feature weights
- **Live preview predictions**: Returns random probability [0-100]
- **Response time**: Fast (<50ms) since no model inference occurs
- **Health check**: Reports `model_loaded=false` to indicate mock mode

## Replacing Mocks with Real Model

To integrate the actual machine learning model from `models/ft.pkl`:

### Step 1: Load the Model

Edit `src/ml/predictor.py` and add model loading logic at the top of the file:

```python
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "ft.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    MODEL_LOADED = False
```

### Step 2: Replace Mock Functions

#### For Tabular Predictions

Replace the `predict_tabular_mock()` function with actual model inference:

```python
def predict_tabular(data: dict) -> dict:
    if not MODEL_LOADED or model is None:
        raise ModelNotLoadedError("Model not loaded")
    
    feature_vector = [
        data["pl_orbper"], data["pl_orbsmax"], data["pl_eqt"],
        data["pl_insol"], data["pl_imppar"], data["pl_trandep"],
        data["pl_trandur"], data["pl_ratdor"], data["pl_ratror"],
        data["st_teff"], data["st_rad"], data["st_mass"],
        data["st_met"], data["st_logg"], data["sy_gmag"],
        data["sy_rmag"], data["sy_imag"], data["sy_zmag"],
        data["sy_jmag"], data["sy_hmag"], data["sy_kmag"]
    ]
    
    probability = model.predict_proba([feature_vector])[0][1]
    
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_names = list(data.keys())
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
    
    confidence = calculate_confidence(probability, feature_vector)
    
    return {
        "predictedValue": float(probability),
        "confidence": float(confidence),
        "attributeWeights": feature_importance
    }
```

#### For Live Preview Predictions

Replace the `predict_live_preview_mock()` function:

```python
def predict_live_preview(data: dict) -> dict:
    if not MODEL_LOADED or model is None:
        raise ModelNotLoadedError("Model not loaded")
    
    feature_vector = [
        data["plTranmid"], data["stPmdec"], data["stTmag"],
        data["stRade"], data["stDist"], data["plRade"]
    ]
    
    probability = model.predict_proba([feature_vector])[0][1]
    
    return {
        "probability": float(probability * 100)
    }
```

### Step 3: Update Health Check

Edit `src/api/routes/health.py` to reflect model loaded state:

```python
from src.ml.predictor import MODEL_LOADED

@router.get("/health")
async def health_check():
    return {
        "data": {
            "status": "healthy",
            "model_loaded": MODEL_LOADED,
            "timestamp": get_timestamp()
        },
        "timestamp": get_timestamp(),
        "requestId": generate_request_id()
    }
```

### Step 4: Handle Feature Importance

If using SHAP values for feature importance:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(feature_vector)

feature_importance = dict(zip(feature_names, shap_values[0]))

top_features = sorted(
    feature_importance.items(), 
    key=lambda x: abs(x[1]), 
    reverse=True
)[:20]
```

## API Endpoints

### Tabular Prediction

**POST** `/api/v1/predict/tabular`

Predicts exoplanet detection probability based on 21 astronomical parameters.

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{
    "pl_orbper": 10.5,
    "pl_orbsmax": 0.06,
    "pl_eqt": 1000,
    "pl_insol": 500,
    "pl_imppar": 0.55,
    "pl_trandep": 0.2,
    "pl_trandur": 4.0,
    "pl_ratdor": 15.0,
    "pl_ratror": 0.1,
    "st_teff": 5700,
    "st_rad": 1.0,
    "st_mass": 0.96,
    "st_met": -0.05,
    "st_logg": 4.45,
    "sy_gmag": 15.0,
    "sy_rmag": 14.4,
    "sy_imag": 14.2,
    "sy_zmag": 14.2,
    "sy_jmag": 12.8,
    "sy_hmag": 12.5,
    "sy_kmag": 12.4
  }'
```

**Example Response:**

```json
{
  "data": {
    "predictedValue": 0.7823,
    "confidence": 0.8945,
    "attributeWeights": {
      "pl_trandep": 0.234,
      "pl_orbper": -0.156,
      "pl_trandur": 0.189,
      "st_teff": 0.145,
      "sy_gmag": -0.178,
      "pl_ratror": 0.167,
      "st_rad": 0.123,
      "pl_insol": 0.112,
      "st_mass": 0.098,
      "sy_jmag": -0.134,
      "sy_hmag": -0.128,
      "sy_kmag": -0.125,
      "pl_eqt": 0.089,
      "st_met": -0.076,
      "pl_ratdor": 0.067,
      "st_logg": -0.054,
      "pl_imppar": -0.043,
      "sy_rmag": -0.123
    }
  },
  "timestamp": "2025-10-06T12:34:56.789000Z",
  "requestId": "12345678-1234-1234-1234-123456789abc"
}
```

### Live Preview Prediction

**POST** `/api/v1/predict/live-preview`

Simplified prediction for interactive demo with 6 normalized parameters.

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/predict/live-preview \
  -H "Content-Type: application/json" \
  -d '{
    "plTranmid": 0.5,
    "stPmdec": 0.5,
    "stTmag": 0.5,
    "stRade": 0.5,
    "stDist": 0.5,
    "plRade": 0.5
  }'
```

**Example Response:**

```json
{
  "data": {
    "probability": 67.5
  },
  "timestamp": "2025-10-06T12:34:56.789000Z",
  "requestId": "12345678-1234-1234-1234-123456789abc"
}
```

### Health Check

**GET** `/api/v1/health`

Check if the API and model are ready.

**Example Request:**

```bash
curl http://localhost:8000/api/v1/health
```

**Example Response:**

```json
{
  "data": {
    "status": "healthy",
    "model_loaded": false,
    "timestamp": "2025-10-06T12:34:56.789000Z"
  },
  "timestamp": "2025-10-06T12:34:56.789000Z",
  "requestId": "12345678-1234-1234-1234-123456789abc"
}
```

## Configuration

### Environment Variables

Configure the server using environment variables:

- `PORT` - Server port (default: 8000)
- `HOST` - Server host (default: 127.0.0.1)
- `CORS_ORIGINS` - Allowed CORS origins (default: *)
- `LOG_LEVEL` - Logging level (default: info)

Example:

```bash
export PORT=8080
export CORS_ORIGINS="http://localhost:3000,https://example.com"
python src/main.py
```

## Error Handling

All errors follow a standardized format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "specific_field",
      "constraint": "validation constraint"
    }
  },
  "timestamp": "2025-10-06T12:34:56.789000Z",
  "requestId": "12345678-1234-1234-1234-123456789abc"
}
```

### Error Codes

- `VALIDATION_ERROR` - Invalid input data (400)
- `MISSING_FIELD` - Required field not provided (400)
- `INVALID_TYPE` - Wrong data type (400)
- `OUT_OF_RANGE` - Value outside allowed range (400)
- `MODEL_ERROR` - Model prediction failed (500)
- `MODEL_NOT_LOADED` - Model not initialized (503)
- `INTERNAL_ERROR` - Unexpected server error (500)

## Testing

### Test Validation Error

```bash
curl -X POST http://localhost:8000/api/v1/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{
    "pl_orbper": 150.0,
    "pl_orbsmax": 0.06
  }'
```

Expected: 400 error with validation details

### Test Missing Field

```bash
curl -X POST http://localhost:8000/api/v1/predict/tabular \
  -H "Content-Type: application/json" \
  -d '{
    "pl_orbper": 10.5
  }'
```

Expected: 400 error with missing field details

## Project Structure

```
backend/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── tabular.py      # Tabular prediction endpoint
│   │   │   ├── live_preview.py # Live preview endpoint
│   │   │   └── health.py       # Health check endpoint
│   │   └── schemas/
│   │       ├── base.py         # Base response wrapper
│   │       ├── tabular.py      # Tabular schemas
│   │       ├── live_preview.py # Live preview schemas
│   │       └── errors.py       # Error schemas
│   ├── ml/
│   │   └── predictor.py        # Prediction logic (MOCK)
│   └── utils/
│       ├── exceptions.py       # Custom exceptions
│       ├── error_handlers.py   # FastAPI exception handlers
│       └── request_utils.py    # Utility functions
├── models/
│   └── ft.pkl                  # ML model file (not used in mock mode)
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

## License

MIT
