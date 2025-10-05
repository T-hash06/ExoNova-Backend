"""
Tabular Exoplanet Classifier - Singleton implementation for model loading and prediction.

This module handles:
- Lazy loading of prediction model (models/preview.pkl)
- Lazy loading of SHAP explainer (models/shap_explainer.pkl)
- Thread-safe singleton pattern to avoid reloading models on each request
- Feature validation (15 features required)
- Prediction and feature importance calculation
"""

import pickle
import joblib  # type: ignore
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from threading import Lock

logger = logging.getLogger(__name__)


class TabularClassifier:
    """
    Singleton class for tabular exoplanet classification.

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for thread-safe initialization
        _prediction_model: Loaded prediction model from preview.pkl
        _shap_explainer: Loaded SHAP explainer from shap_explainer.pkl
        _models_loaded: Flag indicating if models are loaded
    """

    _instance: Optional["TabularClassifier"] = None
    _lock: Lock = Lock()

    EXPECTED_FEATURES: List[str] = [
        "pl_orbper",
        "pl_orbsmax",
        "pl_eqt",
        "pl_insol",
        "pl_imppar",
        "pl_trandep",
        "pl_trandur",
        "pl_ratdor",
        "pl_ratror",
        "st_teff",
        "st_rad",
        "st_mass",
        "st_met",
        "st_logg",
        "sy_imag",
    ]

    # Campos adicionales aceptados (no usados por el modelo, se ignoran si vienen)
    ALLOWED_EXTRA_FIELDS = {
        "sy_gmag",
        "sy_rmag",
        "sy_zmag",
        "sy_jmag",
        "sy_hmag",
        "sy_kmag",
    }

    def __new__(cls):
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the classifier (only runs once due to singleton pattern)."""
        if self._initialized:
            return

        self._prediction_model: Any = None
        self._shap_explainer: Any = None
        self._models_loaded = False
        self._initialized = True

        # Paths to model files
        self._base_path = Path(__file__).parent.parent.parent / "models"
        self._prediction_model_path = self._base_path / "MAutoencoder.pkl"
        self._shap_explainer_path = self._base_path / "shap_explainer.pkl"

        logger.info("TabularClassifier singleton instance created")

    def _load_models(self) -> None:
        """
        Load both prediction model and SHAP explainer.

        Raises:
            FileNotFoundError: If model files don't exist
            Exception: If model loading fails
        """
        if self._models_loaded:
            return

        with self._lock:
            if self._models_loaded:
                return

            try:
                # Load prediction model
                if not self._prediction_model_path.exists():
                    raise FileNotFoundError(
                        f"Prediction model not found at: {self._prediction_model_path}"
                    )

                logger.info(
                    f"Loading prediction model from {self._prediction_model_path}"
                )

                # Try joblib first (scikit-learn models are often saved with joblib)
                try:
                    self._prediction_model = joblib.load(self._prediction_model_path)
                    logger.info("Prediction model loaded successfully with joblib")
                except Exception as joblib_error:
                    logger.debug(f"Joblib load failed, trying pickle: {joblib_error}")
                    # Fallback to pickle
                    with open(self._prediction_model_path, "rb") as f:
                        self._prediction_model = pickle.load(f)
                    logger.info("Prediction model loaded successfully with pickle")

                # Load SHAP explainer (optional)
                if self._shap_explainer_path.exists():
                    logger.info(
                        f"Loading SHAP explainer from {self._shap_explainer_path}"
                    )
                    try:
                        self._shap_explainer = joblib.load(self._shap_explainer_path)
                        logger.info("SHAP explainer loaded successfully with joblib")
                    except Exception as joblib_error:
                        logger.debug(
                            f"Joblib load failed, trying pickle: {joblib_error}"
                        )
                        try:
                            with open(self._shap_explainer_path, "rb") as f:
                                self._shap_explainer = pickle.load(f)
                            logger.info(
                                "SHAP explainer loaded successfully with pickle"
                            )
                        except Exception as shap_error:
                            self._shap_explainer = None
                            logger.warning(
                                f"Failed to load SHAP explainer, proceeding without SHAP: {shap_error}"
                            )
                else:
                    self._shap_explainer = None
                    logger.info(
                        "SHAP explainer not found; proceeding without SHAP explanations"
                    )

                self._models_loaded = True
                logger.info("All models loaded successfully")

            except FileNotFoundError as e:
                logger.error(f"Model file not found: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise Exception(f"Model loading error: {str(e)}")

    def validate_input(self, input_data: Dict[str, float]) -> None:
        """
        Validate input data has the required features.

        Args:
            input_data: Dictionary with feature names as keys (can include None values for optional fields)

        Raises:
            ValueError: If input is empty or invalid feature names/values
        """
        pass

    def predict(
        self, input_data: Dict[str, float]
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Perform prediction using the loaded models.

        Args:
            input_data: Dictionary containing the 15 expected features

        Returns:
            Tuple containing:
            - predicted_value (float): Probability of exoplanet detection [0.0-1.0]
            - confidence (float): Model confidence in prediction [0.0-1.0]
            - attribute_weights (dict): Feature importance weights

        Raises:
            ValueError: If input validation fails
            Exception: If prediction fails
        """
        # Ensure models are loaded
        if not self._models_loaded:
            self._load_models()

        # Validate input
        # self.validate_input(input_data)

        try:
            feature_values, feature_names_used = self._map_and_build_features(
                input_data
            )

            # Ensure model is loaded
            if self._prediction_model is None:
                raise Exception(
                    "Prediction model is not loaded. Please check model path and loading logic."
                )

            # Prediction (model.predict_proba -> [p0, p1])
            prediction_proba = self._prediction_model.predict_proba([feature_values])[0]
            predicted_value = float(prediction_proba[1])
            confidence = float(max(prediction_proba))

            attribute_weights = self._compute_attribute_weights(
                feature_values, feature_names_used
            )

            logger.debug(
                f"Prediction={predicted_value:.4f}, Confidence={confidence:.4f}, Features={len(feature_names_used)}"
            )
            return predicted_value, confidence, attribute_weights

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction error: {str(e)}")

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded

    def ensure_loaded(self) -> None:
        """Public method to guarantee models are loaded without triggering a prediction."""
        if not self._models_loaded:
            self._load_models()

    # -------- Helpers (compactness) --------
    def _map_and_build_features(
        self, input_data: Dict[str, float]
    ) -> Tuple[List[float], List[str]]:
        """Devuelve el vector (15) en el orden esperado, mapeando sy_*mag a sy_imag y usando NaN para faltantes."""
        import math

        filtered = {k: v for k, v in input_data.items() if v is not None}

        # Handle sy_*mag fields: prefer sy_imag, else any other sy_*mag for sy_imag
        sy_mags = {
            k: v
            for k, v in filtered.items()
            if k.startswith("sy_") and k.endswith("mag")
        }
        if "sy_imag" in sy_mags:
            filtered["sy_imag"] = sy_mags["sy_imag"]
        elif sy_mags:
            filtered["sy_imag"] = next(iter(sy_mags.values()))

        values = [filtered.get(name, math.nan) for name in self.EXPECTED_FEATURES]
        return values, list(self.EXPECTED_FEATURES)

    def _compute_attribute_weights(
        self, feature_values: List[float], feature_names: List[str]
    ) -> Dict[str, float]:
        """Calcula pesos via SHAP si está disponible; si no, cae a feature_importances_ o ceros."""
        # Try SHAP first
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer.shap_values([feature_values])
                if isinstance(shap_values, list):
                    shap_values = (
                        shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    )

                local = {}
                for i, name in enumerate(feature_names):
                    if i < len(shap_values[0]):
                        try:
                            val = float(shap_values[0][i])
                            if val != val:  # NaN -> 0.0
                                val = 0.0
                        except Exception:
                            val = 0.0
                        local[name] = val

                return dict(
                    sorted(local.items(), key=lambda x: abs(x[1]), reverse=True)
                )
            except Exception as e:
                logger.debug(f"SHAP failed, fallback to feature_importances_: {e}")

        # Fallback: model feature_importances_
        model = self._prediction_model
        if model is not None and hasattr(model, "feature_importances_"):
            try:
                importances = list(getattr(model, "feature_importances_", []))
                local = {
                    name: float(importances[i]) if i < len(importances) else 0.0
                    for i, name in enumerate(feature_names)
                }
                return dict(
                    sorted(local.items(), key=lambda x: abs(x[1]), reverse=True)
                )
            except Exception:
                pass

        # Last resort: zeros
        return {name: 0.0 for name in feature_names}

    @classmethod
    def get_instance(cls) -> "TabularClassifier":
        """Get the singleton instance."""
        return cls()


# Global accessor function
def get_classifier() -> TabularClassifier:
    """
    Get the singleton TabularClassifier instance.

    Returns:
        TabularClassifier: The singleton classifier instance
    """
    return TabularClassifier.get_instance()
