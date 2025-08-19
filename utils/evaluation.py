"""
Evaluation utilities for sentiment analysis models.
Provides comprehensive metrics and visualization for model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str = 'regression') -> Dict[str, float]:
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: 'regression' or 'classification'
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    if task_type == 'regression':
        # Regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        # Mean Absolute Percentage Error (MAPE)
        non_zero_true = y_true[y_true != 0]
        non_zero_pred = y_pred[y_true != 0]
        if len(non_zero_true) > 0:
            metrics['mape'] = np.mean(np.abs((non_zero_true - non_zero_pred) / non_zero_true)) * 100
        
        # Classification metrics (treating as rounded values)
        y_true_rounded = np.round(y_true).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        # Ensure values are in valid range (1-5)
        y_true_rounded = np.clip(y_true_rounded, 1, 5)
        y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
        
        metrics['accuracy'] = accuracy_score(y_true_rounded, y_pred_rounded)
        
        # Precision, recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_rounded, y_pred_rounded, average=None, labels=[1, 2, 3, 4, 5], zero_division=0
        )
        
        for i, star in enumerate([1, 2, 3, 4, 5]):
            metrics[f'precision_{star}star'] = precision[i] if i < len(precision) else 0.0
            metrics[f'recall_{star}star'] = recall[i] if i < len(recall) else 0.0
            metrics[f'f1_{star}star'] = f1[i] if i < len(f1) else 0.0
        
        # Macro and weighted averages
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_rounded, y_pred_rounded, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
        
    else:  # classification
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                metrics[f'precision_{label}'] = precision[i]
                metrics[f'recall_{label}'] = recall[i]
                metrics[f'f1_{label}'] = f1[i]
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
    
    return metrics


def evaluate_model(model, X_test, y_test, task_type: str = 'regression', model_type: str = 'sklearn') -> Dict[str, Any]:
    """
    Evaluate a trained model comprehensively.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels/values
        task_type: 'regression' or 'classification'
        model_type: 'sklearn' or 'transformers'
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Evaluating {model_type} model for {task_type} task")
    
    # Make predictions
    if model_type == 'sklearn':
        y_pred = model.predict(X_test)
    elif model_type == 'transformers':
        # For transformers model, X_test should be tokenized inputs
        import torch
        model.eval()
        with torch.no_grad():
            outputs = model(**X_test)
            y_pred = outputs.logits.cpu().numpy().flatten()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, task_type)
    
    # Create evaluation report
    evaluation_report = {
        'model_type': model_type,
        'task_type': task_type,
        'test_samples': len(y_test),
        'metrics': metrics,
        'predictions': {
            'y_true': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
            'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
        }
    }
    
    logger.info(f"Model evaluation complete. R² Score: {metrics.get('r2', 'N/A')}, "
                f"MAE: {metrics.get('mae', 'N/A')}, Accuracy: {metrics.get('accuracy', 'N/A')}")
    
    return evaluation_report


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                              save_path: Optional[str] = None, 
                              title: str = "Predictions vs Actual") -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix") -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Path to save the plot
        title: Plot title
    """
    # Round predictions for classification
    if y_pred.dtype == float:
        y_pred_rounded = np.round(y_pred).astype(int)
        y_true_rounded = np.round(y_true).astype(int)
    else:
        y_pred_rounded = y_pred
        y_true_rounded = y_true
    
    # Ensure values are in valid range (1-5)
    y_true_rounded = np.clip(y_true_rounded, 1, 5)
    y_pred_rounded = np.clip(y_pred_rounded, 1, 5)
    
    cm = confusion_matrix(y_true_rounded, y_pred_rounded, labels=[1, 2, 3, 4, 5])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels or [1, 2, 3, 4, 5],
                yticklabels=labels or [1, 2, 3, 4, 5])
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           save_path: Optional[str] = None,
                           title: str = "Model Comparison") -> None:
    """
    Plot comparison of metrics across different models.
    
    Args:
        metrics_dict: Dictionary of model names and their metrics
        save_path: Path to save the plot
        title: Plot title
    """
    if not metrics_dict:
        logger.warning("No metrics to plot")
        return
    
    # Extract common metrics
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    
    common_metrics = ['mae', 'rmse', 'r2', 'accuracy', 'f1_macro']
    metrics_to_plot = [m for m in common_metrics if m in all_metrics]
    
    if not metrics_to_plot:
        logger.warning("No common metrics found for plotting")
        return
    
    # Create DataFrame for plotting
    data = []
    for model_name, model_metrics in metrics_dict.items():
        for metric in metrics_to_plot:
            if metric in model_metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': model_metrics[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Metric', y='Value', hue='Model')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def generate_evaluation_report(evaluation_results: Dict[str, Any], 
                              save_path: str) -> None:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        evaluation_results: Results from evaluate_model
        save_path: Path to save the report
    """
    report_path = Path(save_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = report_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Generate text report
    txt_path = report_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write("SENTIMENT ANALYSIS MODEL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model Type: {evaluation_results['model_type']}\n")
        f.write(f"Task Type: {evaluation_results['task_type']}\n")
        f.write(f"Test Samples: {evaluation_results['test_samples']}\n\n")
        
        f.write("METRICS:\n")
        f.write("-" * 30 + "\n")
        
        metrics = evaluation_results['metrics']
        
        # Regression metrics
        if 'mae' in metrics:
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
        if 'rmse' in metrics:
            f.write(f"Root Mean Square Error (RMSE): {metrics['rmse']:.4f}\n")
        if 'r2' in metrics:
            f.write(f"R² Score: {metrics['r2']:.4f}\n")
        if 'mape' in metrics:
            f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n")
        
        # Classification metrics
        if 'accuracy' in metrics:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        if 'f1_macro' in metrics:
            f.write(f"F1 Score (Macro): {metrics['f1_macro']:.4f}\n")
        if 'f1_weighted' in metrics:
            f.write(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
        
        # Per-class metrics
        f.write("\nPER-CLASS METRICS:\n")
        f.write("-" * 30 + "\n")
        for star in [1, 2, 3, 4, 5]:
            precision_key = f'precision_{star}star'
            recall_key = f'recall_{star}star'
            f1_key = f'f1_{star}star'
            
            if all(key in metrics for key in [precision_key, recall_key, f1_key]):
                f.write(f"{star} Star - Precision: {metrics[precision_key]:.4f}, "
                       f"Recall: {metrics[recall_key]:.4f}, "
                       f"F1: {metrics[f1_key]:.4f}\n")
    
    logger.info(f"Evaluation report saved to {report_path}")


def cross_validate_model(model, X, y, cv_folds: int = 5, 
                        task_type: str = 'regression',
                        scoring: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to cross-validate
        X: Features
        y: Labels/values
        cv_folds: Number of CV folds
        task_type: 'regression' or 'classification'
        scoring: Scoring metric
        
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
    
    logger.info(f"Performing {cv_folds}-fold cross-validation")
    
    # Choose appropriate CV strategy
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        default_scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        default_scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    
    scoring = scoring or default_scoring
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                               return_train_score=True, n_jobs=-1)
    
    # Process results
    results = {
        'cv_folds': cv_folds,
        'task_type': task_type,
        'metrics': {}
    }
    
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        # Convert negative metrics back to positive
        if metric.startswith('neg_'):
            test_scores = -test_scores
            train_scores = -train_scores
            metric = metric[4:]  # Remove 'neg_' prefix
        
        results['metrics'][metric] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores),
            'test_scores': test_scores.tolist(),
            'train_scores': train_scores.tolist()
        }
    
    logger.info("Cross-validation completed")
    return results
