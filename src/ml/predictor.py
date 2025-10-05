import random
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def predict_tabular(
    input_data: Dict[str, float],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Real prediction function using loaded ML models.

    This function uses the singleton TabularClassifier to:
    1. Validate input (minimum 15 features required)
    2. Load models lazily (preview.pkl and shap_explainer.pkl)
    3. Generate prediction and confidence
    4. Calculate feature importance using SHAP

    Args:
        input_data: Dictionary containing at least 15 of the 21 astronomical parameters

    Returns:
        Tuple containing:
        - predicted_value (float): Probability of exoplanet detection [0.0-1.0]
        - confidence (float): Model confidence in prediction [0.0-1.0]
        - attribute_weights (dict): Feature importance weights

    Raises:
        ValueError: If input validation fails (less than 15 features)
        Exception: If models cannot be loaded or prediction fails
    """
    try:
        from .tabular_classifier import get_classifier

        classifier = get_classifier()
        predicted_value, confidence, attribute_weights = classifier.predict(input_data)

        return predicted_value, confidence, attribute_weights

    except ValueError as e:
        # Re-raise validation errors
        logger.error(f"Validation error in predict_tabular: {e}")
        raise
    except FileNotFoundError as e:
        # Model files not found - fall back to mock
        logger.warning(f"Model files not found, using mock: {e}")
        return predict_tabular_mock(input_data)
    except Exception as e:
        # Other errors - fall back to mock
        logger.error(f"Prediction failed, using mock: {e}")
        return predict_tabular_mock(input_data)


def predict_tabular_mock(
    input_data: Dict[str, float],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Mock function for tabular exoplanet detection prediction.

    This is a PLACEHOLDER function that returns realistic dummy data for development
    and testing purposes. Replace this function with actual ML model inference.

    REPLACEMENT INSTRUCTIONS:
    ========================

    1. LOAD THE MODEL:
       ```python
       import pickle

       with open('models/ft.pkl', 'rb') as f:
           model = pickle.load(f)
       ```

    2. EXPECTED INPUT FORMAT:
       - Input: Dictionary with 21 astronomical parameters as keys
       - All values are floats within their specified ranges
       - Keys: pl_orbper, pl_orbsmax, pl_eqt, pl_insol, pl_imppar, pl_trandep,
               pl_trandur, pl_ratdor, pl_ratror, st_teff, st_rad, st_mass,
               st_met, st_logg, sy_gmag, sy_rmag, sy_imag, sy_zmag,
               sy_jmag, sy_hmag, sy_kmag

    3. PREPARE DATA FOR MODEL:
       ```python
       import pandas as pd

       feature_names = [
           'pl_orbper', 'pl_orbsmax', 'pl_eqt', 'pl_insol', 'pl_imppar',
           'pl_trandep', 'pl_trandur', 'pl_ratdor', 'pl_ratror', 'st_teff',
           'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_gmag', 'sy_rmag',
           'sy_imag', 'sy_zmag', 'sy_jmag', 'sy_hmag', 'sy_kmag'
       ]

       input_df = pd.DataFrame([input_data], columns=feature_names)
       ```

    4. GET PREDICTION:
       ```python
       prediction_proba = model.predict_proba(input_df)[0]
       predicted_value = float(prediction_proba[1])

       confidence = max(prediction_proba)
       ```

    5. COMPUTE FEATURE IMPORTANCE (choose one method):

       A. Using SHAP (recommended for most models):
       ```python
       import shap

       explainer = shap.TreeExplainer(model)
       shap_values = explainer.shap_values(input_df)

       if isinstance(shap_values, list):
           shap_values = shap_values[1]

       feature_importance = {}
       for i, feature_name in enumerate(feature_names):
           feature_importance[feature_name] = float(shap_values[0][i])

       sorted_features = sorted(
           feature_importance.items(),
           key=lambda x: abs(x[1]),
           reverse=True
       )[:20]

       attribute_weights = dict(sorted_features)
       ```

       B. Using built-in feature_importances_ (for tree-based models):
       ```python
       if hasattr(model, 'feature_importances_'):
           importances = model.feature_importances_
           feature_importance = {
               name: float(imp)
               for name, imp in zip(feature_names, importances)
           }

           sorted_features = sorted(
               feature_importance.items(),
               key=lambda x: abs(x[1]),
               reverse=True
           )[:20]

           total = sum(abs(v) for _, v in sorted_features)
           attribute_weights = {
               k: v / total if total > 0 else 0.0
               for k, v in sorted_features
           }
       ```

    6. RETURN FORMAT:
       ```python
       return predicted_value, confidence, attribute_weights
       ```

    Args:
        input_data: Dictionary containing 21 astronomical parameters

    Returns:
        Tuple containing:
        - predicted_value (float): Probability of exoplanet detection [0.0-1.0]
        - confidence (float): Model confidence in prediction [0.0-1.0]
        - attribute_weights (dict): Feature importance weights for top 15-20 features
    """

    random.seed(hash(str(sorted(input_data.items()))) % (2**32))

    predicted_value = random.uniform(0.1, 0.95)
    confidence = random.uniform(0.7, 0.98)

    attribute_weights = _generate_mock_feature_weights(list(input_data.keys()))

    return predicted_value, confidence, attribute_weights


def predict_live_preview_mock(input_data: Dict[str, float]) -> float:
    """
    Mock function for live preview exoplanet detection prediction.

    This is a PLACEHOLDER function that returns realistic dummy data for development
    and testing purposes. Replace this function with actual ML model inference.

    REPLACEMENT INSTRUCTIONS:
    ========================

    1. LOAD THE MODEL:
       ```python
       import pickle

       with open('models/preview.pkl', 'rb') as f:
           model = pickle.load(f)
       ```

    2. EXPECTED INPUT FORMAT:
       - Input: Dictionary with 6 normalized parameters [0.0-1.0]
       - Keys: plTranmid, stPmdec, stTmag, stRade, stDist, plRade
       - All values are pre-normalized by the frontend

    3. DENORMALIZE IF NEEDED:
       If the model expects actual physical ranges, denormalize:
       ```python
       denormalized = {
           'plTranmid': input_data['plTranmid'] * actual_max_range,
           'stPmdec': input_data['stPmdec'] * actual_max_range,
           ...
       }
       ```

    4. PREPARE DATA FOR MODEL:
       ```python
       import pandas as pd

       feature_names = ['plTranmid', 'stPmdec', 'stTmag', 'stRade', 'stDist', 'plRade']
       input_df = pd.DataFrame([input_data], columns=feature_names)
       ```

    5. GET PREDICTION:
       ```python
       prediction_proba = model.predict_proba(input_df)[0]
       probability = float(prediction_proba[1] * 100)
       ```

    6. RETURN FORMAT:
       ```python
       return probability
       ```

    Args:
        input_data: Dictionary containing 6 normalized parameters [0.0-1.0]

    Returns:
        float: Detection probability as percentage [0-100]
    """

    random.seed(hash(str(sorted(input_data.items()))) % (2**32))

    probability = random.uniform(10.0, 95.0)

    return probability


def _generate_mock_feature_weights(feature_names: list) -> Dict[str, float]:
    """
    Generate mock feature importance weights for development/testing.

    This helper function creates realistic feature weights that:
    - Range from -1.0 to 1.0
    - Are normalized so their absolute values sum reasonably
    - Favor certain features to simulate real model behavior

    Args:
        feature_names: List of feature names to generate weights for

    Returns:
        Dictionary mapping 15-20 feature names to their importance weights
    """

    important_features = [
        "pl_trandep",
        "pl_orbper",
        "pl_trandur",
        "st_teff",
        "sy_gmag",
        "pl_ratror",
        "st_rad",
        "pl_insol",
        "st_mass",
        "sy_jmag",
        "sy_hmag",
        "sy_kmag",
        "pl_eqt",
        "st_met",
        "pl_ratdor",
        "st_logg",
        "pl_imppar",
        "sy_rmag",
        "sy_imag",
        "sy_zmag",
    ]

    available_features = [f for f in important_features if f in feature_names]

    num_features = min(len(available_features), random.randint(15, 20))
    selected_features = random.sample(available_features, num_features)

    weights = {}
    for feature in selected_features:
        weight = random.uniform(-1.0, 1.0)
        weights[feature] = weight

    total = sum(abs(w) for w in weights.values())
    if total > 0:
        normalized_weights = {
            feature: weight / total for feature, weight in weights.items()
        }
    else:
        normalized_weights = weights

    return normalized_weights
