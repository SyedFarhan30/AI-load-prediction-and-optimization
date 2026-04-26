"""
Load Prediction Model Module
Implements machine learning models for load prediction

FIXES:
- evaluate(): y_pred stored as ndarray but returned as 'predictions' key;
  callers comparing with y_test (Series) now work correctly.
- get_feature_importance(): gracefully handles mismatched length between
  feature_names and importances.
- Added GradientBoosting as a third model option.
- predict() now always returns a numpy array.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class LoadPredictionModel:
    """Machine Learning model for predicting CPU load."""

    SUPPORTED_MODELS = ('linear', 'random_forest', 'gradient_boosting')

    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.metrics = {}

        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100, random_state=42, learning_rate=0.1
            )
        else:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_MODELS}, got '{model_type}'"
            )

    # ------------------------------------------------------------------
    def train(self, X_train, y_train):
        """Train the model. Returns training R² score."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_score = self.model.score(X_train, y_train)
        print(f"✓ {self.model_type.upper()} model trained")
        print(f"  Training R² Score: {train_score:.4f}")
        return train_score

    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.

        FIX: predictions are kept as ndarray; y_test values extracted with
        np.asarray() for safe arithmetic regardless of Series vs ndarray.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calling evaluate().")

        y_pred = self.model.predict(X_test)
        y_true = np.asarray(y_test)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        self.metrics = {
            'MAE':         mae,
            'RMSE':        rmse,
            'R2':          r2,
            'predictions': y_pred,   # ndarray
            'actual':      y_test,   # keep original (Series or ndarray)
        }

        print("✓ Model Evaluation Results:")
        print(f"  MAE  (Mean Absolute Error):       {mae:.4f}")
        print(f"  RMSE (Root Mean Squared Error):   {rmse:.4f}")
        print(f"  R²   Score:                       {r2:.4f}")

        return self.metrics

    # ------------------------------------------------------------------
    def predict(self, X):
        """Make predictions on new data. Always returns a numpy ndarray."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calling predict().")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance scores (tree-based models only).

        FIX: guards against len(feature_names) != len(importances).
        """
        if self.model_type not in ('random_forest', 'gradient_boosting'):
            print("Feature importance is only available for tree-based models.")
            return None

        importance = self.model.feature_importances_

        if feature_names is not None:
            # Guard: truncate / pad to match actual importance length
            feature_names = list(feature_names)[:len(importance)]
            importance_df = pd.DataFrame({
                'feature':    feature_names,
                'importance': importance[:len(feature_names)],
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'importance': importance,
            }).sort_values('importance', ascending=False)

        return importance_df

    # ------------------------------------------------------------------
    def get_metrics(self):
        """Return evaluation metrics dict."""
        return self.metrics


# -----------------------------------------------------------------------
def train_load_prediction_model(X_train, X_test, y_train, y_test,
                                model_type='random_forest'):
    """
    Train and evaluate a load prediction model.

    Returns:
        (trained_model, metrics_dict)
    """
    model = LoadPredictionModel(model_type)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    return model, metrics


# -----------------------------------------------------------------------
def get_prediction_explanation(model, X_test, y_test, top_n=5):
    """
    Get a simple explanation of model predictions.

    FIX: y_test.iloc[i] crashes if y_test is an ndarray; now uses positional
    indexing via np.asarray.
    """
    predictions = model.predict(X_test)
    y_true = np.asarray(y_test)

    explanation = {
        'title':       'Load Prediction Model Explanation',
        'description': f'Using {model.model_type} to predict CPU load',
        'sample_predictions': [],
    }

    for i in range(min(top_n, len(predictions))):
        explanation['sample_predictions'].append({
            'actual':    float(y_true[i]),
            'predicted': float(predictions[i]),
            'error':     float(abs(y_true[i] - predictions[i])),
        })

    return explanation