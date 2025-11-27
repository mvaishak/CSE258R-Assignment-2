"""
Utility Functions for DiamondHood
Helper functions for data processing, visualization, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


def format_price(price: float) -> str:
    """
    Format price as currency string.
    
    Args:
        price: Price value
        
    Returns:
        Formatted string (e.g., "$1,234.56")
    """
    return f"${price:,.2f}"


def calculate_price_per_carat(price: float, carat: float) -> float:
    """
    Calculate price per carat.
    
    Args:
        price: Diamond price
        carat: Carat weight
        
    Returns:
        Price per carat
    """
    return price / carat if carat > 0 else 0


def categorize_price(price: float) -> str:
    """
    Categorize diamond into price range.
    
    Args:
        price: Diamond price
        
    Returns:
        Category string
    """
    if price < 1000:
        return "Budget (<$1K)"
    elif price < 3000:
        return "Mid-Range ($1K-$3K)"
    elif price < 10000:
        return "Premium ($3K-$10K)"
    else:
        return "Luxury (>$10K)"


def get_quality_score(cut: str, color: str, clarity: str) -> float:
    """
    Calculate overall quality score from cut, color, clarity.
    
    Args:
        cut: Cut grade
        color: Color grade
        clarity: Clarity grade
        
    Returns:
        Quality score (0-100)
    """
    cut_scores = {'Fair': 20, 'Good': 40, 'Very Good': 60, 'Premium': 80, 'Ideal': 100}
    color_scores = {'J': 14, 'I': 29, 'H': 43, 'G': 57, 'F': 71, 'E': 86, 'D': 100}
    clarity_scores = {'I1': 13, 'SI2': 25, 'SI1': 38, 'VS2': 50, 'VS1': 63, 'VVS2': 75, 'VVS1': 88, 'IF': 100}
    
    cut_score = cut_scores.get(cut, 50)
    color_score = color_scores.get(color, 50)
    clarity_score = clarity_scores.get(clarity, 50)
    
    # Weighted average: cut 40%, clarity 35%, color 25%
    overall_score = (cut_score * 0.4) + (clarity_score * 0.35) + (color_score * 0.25)
    
    return overall_score


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        data: Pandas Series
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Args:
        data: Pandas Series
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Boolean Series indicating outliers
    """
    from scipy import stats
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold


def normalize_features(data: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0, 1] range.
    
    Args:
        data: Input array
        
    Returns:
        Normalized array
    """
    min_val = data.min()
    max_val = data.max()
    
    if max_val - min_val == 0:
        return np.zeros_like(data)
    
    return (data - min_val) / (max_val - min_val)


def standardize_features(data: np.ndarray) -> np.ndarray:
    """
    Z-score standardization (mean=0, std=1).
    
    Args:
        data: Input array
        
    Returns:
        Standardized array
    """
    mean = data.mean()
    std = data.std()
    
    if std == 0:
        return np.zeros_like(data)
    
    return (data - mean) / std


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual"):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot residuals distribution.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residuals Distribution')
    axes[0].grid(alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_metrics_to_json(metrics: Dict, filepath: str):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {filepath}")


def load_metrics_from_json(filepath: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    print(f"✓ Metrics loaded: {filepath}")
    return metrics


def create_feature_importance_plot(feature_names: List[str], importances: np.ndarray, top_n: int = 10):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to show
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances, color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_model_summary(model, X_train, y_train, X_test, y_test):
    """
    Print comprehensive model summary.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    """
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, train_pred)
    
    print("\nTraining Performance:")
    for metric, value in train_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, test_pred)
    
    print("\nTest Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n" + "="*60)


def log_experiment(experiment_name: str, params: Dict, metrics: Dict, log_file: str = "experiments.log"):
    """
    Log experiment parameters and results.
    
    Args:
        experiment_name: Name of experiment
        params: Parameter dictionary
        metrics: Metrics dictionary
        log_file: Path to log file
    """
    from datetime import datetime
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment': experiment_name,
        'parameters': params,
        'metrics': metrics
    }
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"✓ Experiment logged: {experiment_name}")


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table for multiple models.
    
    Args:
        results: Dictionary of {model_name: {metric: value}}
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by R2 score if available
    if 'r2' in df.columns:
        df = df.sort_values('r2', ascending=False)
    
    return df


if __name__ == "__main__":
    print("DiamondHood Utilities Loaded")
    print("\nAvailable functions:")
    print("  - format_price: Format currency")
    print("  - calculate_price_per_carat: Price per carat")
    print("  - get_quality_score: Overall quality score")
    print("  - detect_outliers_iqr: IQR outlier detection")
    print("  - calculate_regression_metrics: Model evaluation")
    print("  - plot_prediction_vs_actual: Visualization")
    print("  - save/load_metrics_to_json: Persistence")
    print("  - compare_models: Model comparison table")
