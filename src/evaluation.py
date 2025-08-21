"""
Evaluation utilities for sentiment analysis models and predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import json
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentEvaluationMetrics:
    """
    Comprehensive evaluation metrics for sentiment analysis.
    """
    
    @staticmethod
    def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Calculate regression metrics for continuous sentiment scores.
        
        Args:
            y_true: True sentiment scores
            y_pred: Predicted sentiment scores
            
        Returns:
            Dictionary with regression metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Custom accuracy metrics
        accuracy_0_5 = np.mean(np.abs(y_true - y_pred) <= 0.5)
        accuracy_1_0 = np.mean(np.abs(y_true - y_pred) <= 1.0)
        
        # Pearson correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        
        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "accuracy_0.5": round(accuracy_0_5, 4),
            "accuracy_1.0": round(accuracy_1_0, 4),
            "correlation": round(correlation, 4)
        }
    
    @staticmethod
    def classification_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
        """
        Calculate classification metrics by converting scores to classes.
        
        Args:
            y_true: True sentiment scores
            y_pred: Predicted sentiment scores
            
        Returns:
            Dictionary with classification metrics
        """
        def score_to_class(scores):
            classes = []
            for score in scores:
                if score <= 2.0:
                    classes.append("negative")
                elif score >= 4.0:
                    classes.append("positive")
                else:
                    classes.append("neutral")
            return classes
        
        y_true_classes = score_to_class(y_true)
        y_pred_classes = score_to_class(y_pred)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average=None, labels=["negative", "neutral", "positive"]
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average="macro"
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average="weighted"
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=["negative", "neutral", "positive"])
        
        return {
            "accuracy": round(accuracy, 4),
            "macro_avg": {
                "precision": round(precision_macro, 4),
                "recall": round(recall_macro, 4),
                "f1": round(f1_macro, 4)
            },
            "weighted_avg": {
                "precision": round(precision_weighted, 4),
                "recall": round(recall_weighted, 4),
                "f1": round(f1_weighted, 4)
            },
            "per_class": {
                "negative": {
                    "precision": round(precision[0], 4),
                    "recall": round(recall[0], 4),
                    "f1": round(f1[0], 4),
                    "support": int(support[0])
                },
                "neutral": {
                    "precision": round(precision[1], 4),
                    "recall": round(recall[1], 4),
                    "f1": round(f1[1], 4),
                    "support": int(support[1])
                },
                "positive": {
                    "precision": round(precision[2], 4),
                    "recall": round(recall[2], 4),
                    "f1": round(f1[2], 4),
                    "support": int(support[2])
                }
            },
            "confusion_matrix": cm.tolist()
        }
    
    @staticmethod
    def error_analysis(y_true: List[float], y_pred: List[float], 
                      texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors in detail.
        
        Args:
            y_true: True sentiment scores
            y_pred: Predicted sentiment scores
            texts: Optional list of corresponding texts
            
        Returns:
            Dictionary with error analysis
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        errors = np.abs(y_true - y_pred)
        
        # Error statistics
        error_stats = {
            "mean_error": round(np.mean(errors), 4),
            "median_error": round(np.median(errors), 4),
            "std_error": round(np.std(errors), 4),
            "max_error": round(np.max(errors), 4),
            "min_error": round(np.min(errors), 4)
        }
        
        # Error distribution by score ranges
        score_ranges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        range_analysis = {}
        
        for low, high in score_ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                range_analysis[f"{low}-{high}"] = {
                    "count": int(mask.sum()),
                    "mean_error": round(np.mean(errors[mask]), 4),
                    "std_error": round(np.std(errors[mask]), 4),
                    "mean_true": round(np.mean(y_true[mask]), 4),
                    "mean_pred": round(np.mean(y_pred[mask]), 4)
                }
        
        # Top errors
        top_error_indices = np.argsort(errors)[-10:][::-1]
        top_errors = []
        
        for idx in top_error_indices:
            error_info = {
                "index": int(idx),
                "true_score": round(y_true[idx], 4),
                "predicted_score": round(y_pred[idx], 4),
                "error": round(errors[idx], 4)
            }
            
            if texts is not None and idx < len(texts):
                error_info["text"] = texts[idx]
            
            top_errors.append(error_info)
        
        # Bias analysis
        bias = np.mean(y_pred - y_true)
        
        return {
            "error_statistics": error_stats,
            "bias": round(bias, 4),
            "error_by_score_range": range_analysis,
            "top_errors": top_errors
        }


class SentimentEvaluationReport:
    """
    Generate comprehensive evaluation reports for sentiment analysis models.
    """
    
    def __init__(self, model_name: str = "Sentiment Model"):
        """
        Initialize evaluation report generator.
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics_calculator = SentimentEvaluationMetrics()
    
    def evaluate_predictions(self, y_true: List[float], y_pred: List[float],
                           texts: Optional[List[str]] = None,
                           confidences: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True sentiment scores
            y_pred: Predicted sentiment scores
            texts: Optional list of corresponding texts
            confidences: Optional list of prediction confidences
            
        Returns:
            Complete evaluation report
        """
        # Basic metrics
        regression_metrics = self.metrics_calculator.regression_metrics(y_true, y_pred)
        classification_metrics = self.metrics_calculator.classification_metrics(y_true, y_pred)
        error_analysis = self.metrics_calculator.error_analysis(y_true, y_pred, texts)
        
        # Confidence analysis
        confidence_analysis = None
        if confidences is not None:
            confidence_analysis = self._analyze_confidence(y_true, y_pred, confidences)
        
        # Generate report
        report = {
            "model_name": self.model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "num_samples": len(y_true),
                "score_range": {
                    "min": round(min(y_true), 2),
                    "max": round(max(y_true), 2),
                    "mean": round(np.mean(y_true), 2),
                    "std": round(np.std(y_true), 2)
                }
            },
            "regression_metrics": regression_metrics,
            "classification_metrics": classification_metrics,
            "error_analysis": error_analysis,
            "confidence_analysis": confidence_analysis
        }
        
        return report
    
    def _analyze_confidence(self, y_true: List[float], y_pred: List[float], 
                          confidences: List[float]) -> Dict[str, Any]:
        """
        Analyze relationship between prediction confidence and accuracy.
        
        Args:
            y_true: True sentiment scores
            y_pred: Predicted sentiment scores
            confidences: Prediction confidences
            
        Returns:
            Confidence analysis results
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        confidences = np.array(confidences)
        errors = np.abs(y_true - y_pred)
        
        # Confidence statistics
        conf_stats = {
            "mean_confidence": round(np.mean(confidences), 4),
            "std_confidence": round(np.std(confidences), 4),
            "min_confidence": round(np.min(confidences), 4),
            "max_confidence": round(np.max(confidences), 4)
        }
        
        # Correlation between confidence and accuracy
        accuracy_by_confidence = 1 / (1 + errors)  # Higher accuracy = lower error
        conf_accuracy_corr = np.corrcoef(confidences, accuracy_by_confidence)[0, 1]
        
        # Binned analysis
        conf_bins = np.linspace(0, 1, 6)  # 5 bins
        bin_analysis = {}
        
        for i in range(len(conf_bins) - 1):
            low, high = conf_bins[i], conf_bins[i + 1]
            mask = (confidences >= low) & (confidences < high)
            
            if mask.sum() > 0:
                bin_analysis[f"{low:.1f}-{high:.1f}"] = {
                    "count": int(mask.sum()),
                    "mean_error": round(np.mean(errors[mask]), 4),
                    "mean_confidence": round(np.mean(confidences[mask]), 4),
                    "accuracy_0.5": round(np.mean(errors[mask] <= 0.5), 4)
                }
        
        return {
            "confidence_statistics": conf_stats,
            "confidence_accuracy_correlation": round(conf_accuracy_corr, 4),
            "confidence_bins": bin_analysis
        }
    
    def generate_visualization_data(self, y_true: List[float], y_pred: List[float],
                                  confidences: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate data for creating visualizations.
        
        Args:
            y_true: True sentiment scores
            y_pred: Predicted sentiment scores
            confidences: Optional prediction confidences
            
        Returns:
            Data for visualizations
        """
        # Scatter plot data
        scatter_data = {
            "true_scores": y_true,
            "predicted_scores": y_pred,
            "confidences": confidences
        }
        
        # Error distribution
        errors = np.abs(np.array(y_true) - np.array(y_pred))
        error_hist = np.histogram(errors, bins=20)
        
        error_distribution = {
            "bin_edges": error_hist[1].tolist(),
            "counts": error_hist[0].tolist()
        }
        
        # Score distribution comparison
        true_hist = np.histogram(y_true, bins=np.arange(1, 6.1, 0.5))
        pred_hist = np.histogram(y_pred, bins=np.arange(1, 6.1, 0.5))
        
        score_distribution = {
            "bin_edges": true_hist[1].tolist(),
            "true_counts": true_hist[0].tolist(),
            "pred_counts": pred_hist[0].tolist()
        }
        
        # Confusion matrix data (for 3-class classification)
        def score_to_class_idx(scores):
            classes = []
            for score in scores:
                if score <= 2.0:
                    classes.append(0)  # negative
                elif score >= 4.0:
                    classes.append(2)  # positive
                else:
                    classes.append(1)  # neutral
            return classes
        
        true_classes = score_to_class_idx(y_true)
        pred_classes = score_to_class_idx(y_pred)
        cm = confusion_matrix(true_classes, pred_classes, labels=[0, 1, 2])
        
        confusion_data = {
            "matrix": cm.tolist(),
            "labels": ["Negative", "Neutral", "Positive"]
        }
        
        return {
            "scatter_plot": scatter_data,
            "error_distribution": error_distribution,
            "score_distribution": score_distribution,
            "confusion_matrix": confusion_data
        }
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """
        Save evaluation report to JSON file.
        
        Args:
            report: Evaluation report dictionary
            filepath: Path to save the report
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """
        Print a summary of the evaluation report.
        
        Args:
            report: Evaluation report dictionary
        """
        print(f"\n{'='*60}")
        print(f"SENTIMENT ANALYSIS EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"Model: {report['model_name']}")
        print(f"Samples: {report['dataset_info']['num_samples']:,}")
        print(f"Evaluation Time: {report['evaluation_timestamp']}")
        
        print(f"\n{'Regression Metrics:':-^40}")
        reg_metrics = report['regression_metrics']
        print(f"MAE (Mean Absolute Error): {reg_metrics['mae']:.4f}")
        print(f"RMSE (Root Mean Square Error): {reg_metrics['rmse']:.4f}")
        print(f"R² Score: {reg_metrics['r2']:.4f}")
        print(f"Accuracy (±0.5 stars): {reg_metrics['accuracy_0.5']:.4f}")
        print(f"Accuracy (±1.0 stars): {reg_metrics['accuracy_1.0']:.4f}")
        print(f"Correlation: {reg_metrics['correlation']:.4f}")
        
        print(f"\n{'Classification Metrics:':-^40}")
        class_metrics = report['classification_metrics']
        print(f"Overall Accuracy: {class_metrics['accuracy']:.4f}")
        print(f"Macro F1-Score: {class_metrics['macro_avg']['f1']:.4f}")
        print(f"Weighted F1-Score: {class_metrics['weighted_avg']['f1']:.4f}")
        
        print(f"\n{'Per-Class Performance:':-^40}")
        for class_name, metrics in class_metrics['per_class'].items():
            print(f"{class_name.capitalize():>8}: P={metrics['precision']:.3f} "
                  f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                  f"(n={metrics['support']})")
        
        print(f"\n{'Error Analysis:':-^40}")
        error_analysis = report['error_analysis']
        print(f"Mean Error: {error_analysis['error_statistics']['mean_error']:.4f}")
        print(f"Max Error: {error_analysis['error_statistics']['max_error']:.4f}")
        print(f"Bias: {error_analysis['bias']:.4f}")
        
        if report['confidence_analysis']:
            print(f"\n{'Confidence Analysis:':-^40}")
            conf_analysis = report['confidence_analysis']
            print(f"Mean Confidence: {conf_analysis['confidence_statistics']['mean_confidence']:.4f}")
            print(f"Confidence-Accuracy Correlation: {conf_analysis['confidence_accuracy_correlation']:.4f}")
        
        print(f"\n{'='*60}")


class ModelComparisonEvaluator:
    """
    Compare multiple sentiment analysis models.
    """
    
    def __init__(self):
        """Initialize model comparison evaluator."""
        self.metrics_calculator = SentimentEvaluationMetrics()
    
    def compare_models(self, model_predictions: Dict[str, List[float]], 
                      y_true: List[float]) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            model_predictions: Dictionary of {model_name: predictions}
            y_true: True sentiment scores
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model_name, y_pred in model_predictions.items():
            reg_metrics = self.metrics_calculator.regression_metrics(y_true, y_pred)
            class_metrics = self.metrics_calculator.classification_metrics(y_true, y_pred)
            
            comparison_results[model_name] = {
                "mae": reg_metrics["mae"],
                "rmse": reg_metrics["rmse"],
                "r2": reg_metrics["r2"],
                "accuracy_0.5": reg_metrics["accuracy_0.5"],
                "class_accuracy": class_metrics["accuracy"],
                "macro_f1": class_metrics["macro_avg"]["f1"]
            }
        
        # Rank models by different metrics
        rankings = {}
        for metric in ["mae", "rmse", "r2", "accuracy_0.5", "class_accuracy", "macro_f1"]:
            if metric in ["mae", "rmse"]:  # Lower is better
                rankings[metric] = sorted(model_predictions.keys(), 
                                        key=lambda x: comparison_results[x][metric])
            else:  # Higher is better
                rankings[metric] = sorted(model_predictions.keys(), 
                                        key=lambda x: comparison_results[x][metric], 
                                        reverse=True)
        
        return {
            "model_metrics": comparison_results,
            "rankings": rankings,
            "best_overall": self._determine_best_model(comparison_results)
        }
    
    def _determine_best_model(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Determine the best model based on weighted score.
        
        Args:
            results: Model comparison results
            
        Returns:
            Name of the best model
        """
        # Weights for different metrics (higher weight = more important)
        weights = {
            "mae": -0.3,  # Negative because lower is better
            "rmse": -0.2,
            "r2": 0.2,
            "accuracy_0.5": 0.25,
            "class_accuracy": 0.15,
            "macro_f1": 0.1
        }
        
        model_scores = {}
        
        for model_name, metrics in results.items():
            score = sum(weights[metric] * metrics[metric] for metric in weights.keys())
            model_scores[model_name] = score
        
        return max(model_scores.keys(), key=lambda x: model_scores[x])


if __name__ == "__main__":
    # Example usage
    import random
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True scores
    y_true = np.random.uniform(1, 5, n_samples)
    
    # Predicted scores with some noise
    y_pred = y_true + np.random.normal(0, 0.3, n_samples)
    y_pred = np.clip(y_pred, 1, 5)
    
    # Sample texts
    texts = [f"Sample text {i}" for i in range(n_samples)]
    
    # Sample confidences
    confidences = np.random.uniform(0.2, 0.9, n_samples)
    
    # Create evaluation report
    evaluator = SentimentEvaluationReport("Test Model")
    report = evaluator.evaluate_predictions(y_true, y_pred, texts, confidences)
    
    # Print summary
    evaluator.print_summary(report)
    
    # Save report
    evaluator.save_report(report, "test_evaluation_report.json")
    
    print("\nEvaluation completed successfully!")
