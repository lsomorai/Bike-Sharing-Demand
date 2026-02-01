"""Model evaluation utilities for bike sharing demand prediction."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return performance metrics.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict method
    X_test : array-like
        Test features
    y_test : array-like
        True target values
    model_name : str
        Name for display purposes

    Returns
    -------
    dict
        Dictionary containing r2, rmse, and mae scores
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    print(f"\n{model_name} Performance:")
    print(f"  R² Score: {metrics['r2']:.4f}")
    print(f"  RMSE:     {metrics['rmse']:.2f}")
    print(f"  MAE:      {metrics['mae']:.2f}")

    return metrics


def plot_residuals(y_test, y_pred, model_name="Model", ax=None):
    """
    Plot residuals vs predicted values.

    Parameters
    ----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted values
    model_name : str
        Name for plot title
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="none")
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{model_name} - Residual Plot")
    ax.grid(True, alpha=0.3)

    return ax


def plot_predictions(y_test, y_pred, model_name="Model", ax=None):
    """
    Plot predicted vs actual values.

    Parameters
    ----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted values
    model_name : str
        Name for plot title
    ax : matplotlib axis, optional
        Axis to plot on

    Returns
    -------
    matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="none")

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_name} - Predicted vs Actual")
    ax.grid(True, alpha=0.3)

    return ax


def save_model(model, filepath):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : sklearn estimator
        Trained model to save
    filepath : str
        Path to save the model (should end in .joblib)
    """
    import joblib
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model

    Returns
    -------
    sklearn estimator
        Loaded model
    """
    import joblib
    return joblib.load(filepath)
