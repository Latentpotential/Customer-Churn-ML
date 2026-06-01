"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import glob
import json
import pandas as pd
import mlflow

# Project root = two levels up from this file (src/serving/inference.py -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _discover_model_dir() -> str:
    """
    Resolve the MLflow model directory, in priority order:

    1. MODEL_DIR env var (explicit override, e.g. set in Docker).
    2. /app/model (the containerized path used in production).
    3. Newest model under the MLflow 3.x layout: mlruns/<exp>/models/m-<hash>/artifacts.
    4. Legacy MLflow layout: mlruns/<exp>/<run>/artifacts/model.

    Returns the first path that exists. Raises if none are found.
    """
    candidates = []

    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        candidates.append(env_dir)

    candidates.append("/app/model")

    # MLflow 3.x logged-model layout (newest first)
    new_layout = glob.glob(os.path.join(_PROJECT_ROOT, "mlruns", "*", "models", "*", "artifacts"))
    new_layout = [p for p in new_layout if os.path.exists(os.path.join(p, "MLmodel"))]
    new_layout.sort(key=os.path.getmtime, reverse=True)
    candidates.extend(new_layout)

    # Legacy layout fallback
    legacy = glob.glob(os.path.join(_PROJECT_ROOT, "mlruns", "*", "*", "artifacts", "model"))
    legacy.sort(key=os.path.getmtime, reverse=True)
    candidates.extend(legacy)

    for path in candidates:
        if path and os.path.exists(os.path.join(path, "MLmodel")):
            return path

    raise FileNotFoundError(
        "No MLflow model found. Set MODEL_DIR, or train one first:\n"
        "  python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn"
    )


def _load_feature_cols(model_dir: str) -> list:
    """
    Load the exact training feature-column order, in priority order:

    1. FEATURE_COLUMNS_PATH env var (explicit override).
    2. <model_dir>/feature_columns.txt  (production: copied next to the model).
    3. artifacts/feature_columns.json    (local dev: written by run_pipeline.py).

    CRITICAL: this order must match training, or the model gets mis-aligned features.
    """
    # 1. explicit override (.txt newline-separated)
    env_path = os.environ.get("FEATURE_COLUMNS_PATH")
    if env_path and os.path.exists(env_path):
        with open(env_path) as f:
            return [ln.strip() for ln in f if ln.strip()]

    # 2. alongside the model (Docker production path)
    txt_path = os.path.join(model_dir, "feature_columns.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return [ln.strip() for ln in f if ln.strip()]

    # 3. local artifacts dir written by the training pipeline (JSON list)
    json_path = os.path.join(_PROJECT_ROOT, "artifacts", "feature_columns.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)

    raise FileNotFoundError(
        "Could not find feature columns. Expected one of:\n"
        f"  - $FEATURE_COLUMNS_PATH\n  - {txt_path}\n  - {json_path}\n"
        "Run scripts/run_pipeline.py to generate artifacts/feature_columns.json."
    )


# === MODEL LOADING ===
MODEL_DIR = _discover_model_dir()
model = mlflow.pyfunc.load_model(MODEL_DIR)
print(f"✅ Model loaded successfully from {MODEL_DIR}")

# === FEATURE SCHEMA LOADING ===
FEATURE_COLS = _load_feature_cols(MODEL_DIR)
print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service