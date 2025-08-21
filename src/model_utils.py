"""
Model utilities for sentiment analysis inference and evaluation.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModelConfig:
    """Configuration class for sentiment analysis model."""
    
    def __init__(self, model_path: str):
        """
        Initialize configuration from model directory.
        
        Args:
            model_path: Path to the model directory
        """
        self.model_path = Path(model_path)
        self.config_path = self.model_path / "config.json"
        self.model_info_path = self.model_path / "model_info.json"
        
        # Load configuration
        self.config = self._load_config()
        self.model_info = self._load_model_info()
        
        # Extract key parameters
        self.max_length = self.model_info.get("training_info", {}).get("max_length", 512)
        self.model_name = self.model_info.get("training_info", {}).get("model_name", "roberta-base")
        self.num_labels = self.config.get("num_labels", 1)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from config.json."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config.json: {e}")
            return {}
    
    def _load_model_info(self) -> Dict[str, Any]:
        """Load model information from model_info.json."""
        try:
            with open(self.model_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model_info.json: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics."""
        test_metrics = self.model_info.get("test_metrics", {})
        return {
            "accuracy": test_metrics.get("eval_accuracy", 0.0),
            "mae": test_metrics.get("eval_mae", 0.0),
            "rmse": test_metrics.get("eval_rmse", 0.0),
            "r2": test_metrics.get("eval_r2", 0.0),
            "loss": test_metrics.get("eval_loss", 0.0)
        }


class SentimentPredictor:
    """
    Sentiment analysis predictor with score denormalization and confidence estimation.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the sentiment predictor.
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self.config = SentimentModelConfig(model_path)
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"Sentiment predictor initialized on {self.device}")
        logger.info(f"Model metrics: {self.config.get_metrics()}")
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=self.config.num_labels
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        return inputs
    
    def _denormalize_score(self, normalized_score: float) -> float:
        """
        Convert normalized model output [0,1] back to sentiment score [1,5].
        
        Args:
            normalized_score: Normalized score from model
            
        Returns:
            Denormalized sentiment score
        """
        # Model was trained with labels normalized from [1,5] to [0,1]
        # Denormalize: score = normalized * (max - min) + min
        denormalized = normalized_score * 4.0 + 1.0
        
        # Ensure score is within valid range
        return np.clip(denormalized, 1.0, 5.0)
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with prediction results
        """
        if not text or not text.strip():
            return {
                "sentiment_score": 3.0,
                "confidence": 0.0,
                "normalized_score": 0.5,
                "prediction_class": "neutral"
            }
        
        try:
            # Preprocess text
            inputs = self._preprocess_text(text.strip())
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply sigmoid for regression output
                normalized_score = torch.sigmoid(logits).item()
            
            # Denormalize score
            sentiment_score = self._denormalize_score(normalized_score)
            
            # Calculate confidence (distance from 0.5 in normalized space)
            confidence = abs(normalized_score - 0.5) * 2.0
            
            # Determine prediction class
            if sentiment_score <= 2.0:
                prediction_class = "negative"
            elif sentiment_score >= 4.0:
                prediction_class = "positive"
            else:
                prediction_class = "neutral"
            
            return {
                "sentiment_score": round(sentiment_score, 3),
                "confidence": round(confidence, 3),
                "normalized_score": round(normalized_score, 3),
                "prediction_class": prediction_class
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                "sentiment_score": 3.0,
                "confidence": 0.0,
                "normalized_score": 0.5,
                "prediction_class": "neutral",
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input text strings
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # Make predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply sigmoid for regression output
                    normalized_scores = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # Handle single prediction case
                if normalized_scores.ndim == 0:
                    normalized_scores = [normalized_scores.item()]
                
                # Process each prediction
                for j, normalized_score in enumerate(normalized_scores):
                    sentiment_score = self._denormalize_score(normalized_score)
                    confidence = abs(normalized_score - 0.5) * 2.0
                    
                    # Determine prediction class
                    if sentiment_score <= 2.0:
                        prediction_class = "negative"
                    elif sentiment_score >= 4.0:
                        prediction_class = "positive"
                    else:
                        prediction_class = "neutral"
                    
                    batch_results.append({
                        "sentiment_score": round(sentiment_score, 3),
                        "confidence": round(confidence, 3),
                        "normalized_score": round(normalized_score, 3),
                        "prediction_class": prediction_class
                    })
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                # Add error results for this batch
                for _ in batch_texts:
                    results.append({
                        "sentiment_score": 3.0,
                        "confidence": 0.0,
                        "normalized_score": 0.5,
                        "prediction_class": "neutral",
                        "error": str(e)
                    })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and performance metrics
        """
        metrics = self.config.get_metrics()
        training_info = self.config.model_info.get("training_info", {})
        
        return {
            "model_name": self.config.model_name,
            "model_path": str(self.model_path),
            "max_length": self.config.max_length,
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "performance_metrics": metrics,
            "training_info": {
                "training_samples": training_info.get("training_samples", 0),
                "validation_samples": training_info.get("validation_samples", 0),
                "test_samples": training_info.get("test_samples", 0),
                "num_epochs": training_info.get("num_epochs", 0),
                "learning_rate": training_info.get("learning_rate", 0),
                "batch_size": training_info.get("batch_size", 0)
            }
        }


class SentimentEvaluator:
    """
    Evaluator for sentiment analysis models.
    """
    
    def __init__(self, predictor: SentimentPredictor):
        """
        Initialize evaluator with a predictor.
        
        Args:
            predictor: SentimentPredictor instance
        """
        self.predictor = predictor
    
    def evaluate_predictions(self, texts: List[str], true_scores: List[float]) -> Dict[str, float]:
        """
        Evaluate model predictions against ground truth.
        
        Args:
            texts: List of input texts
            true_scores: List of true sentiment scores
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Get predictions
        predictions = self.predictor.predict_batch(texts)
        predicted_scores = [pred["sentiment_score"] for pred in predictions]
        
        # Calculate metrics
        mae = mean_absolute_error(true_scores, predicted_scores)
        mse = mean_squared_error(true_scores, predicted_scores)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_scores, predicted_scores)
        
        # Calculate accuracy (within 0.5 stars)
        accuracy = np.mean(np.abs(np.array(true_scores) - np.array(predicted_scores)) <= 0.5)
        
        # Calculate class-wise accuracy
        def score_to_class(score):
            if score <= 2.0:
                return "negative"
            elif score >= 4.0:
                return "positive"
            else:
                return "neutral"
        
        true_classes = [score_to_class(score) for score in true_scores]
        pred_classes = [pred["prediction_class"] for pred in predictions]
        class_accuracy = np.mean(np.array(true_classes) == np.array(pred_classes))
        
        return {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "accuracy_0.5": round(accuracy, 4),
            "class_accuracy": round(class_accuracy, 4),
            "num_samples": len(texts)
        }
    
    def analyze_errors(self, texts: List[str], true_scores: List[float], 
                      top_errors: int = 10) -> Dict[str, Any]:
        """
        Analyze prediction errors to identify patterns.
        
        Args:
            texts: List of input texts
            true_scores: List of true sentiment scores
            top_errors: Number of top errors to return
            
        Returns:
            Dictionary with error analysis
        """
        # Get predictions
        predictions = self.predictor.predict_batch(texts)
        predicted_scores = [pred["sentiment_score"] for pred in predictions]
        
        # Calculate errors
        errors = np.abs(np.array(true_scores) - np.array(predicted_scores))
        
        # Find worst predictions
        worst_indices = np.argsort(errors)[-top_errors:][::-1]
        
        worst_predictions = []
        for idx in worst_indices:
            worst_predictions.append({
                "text": texts[idx],
                "true_score": true_scores[idx],
                "predicted_score": predicted_scores[idx],
                "error": round(errors[idx], 3),
                "confidence": predictions[idx]["confidence"]
            })
        
        # Error distribution by score range
        score_ranges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        range_errors = {}
        
        for low, high in score_ranges:
            mask = (np.array(true_scores) >= low) & (np.array(true_scores) < high)
            if mask.sum() > 0:
                range_errors[f"{low}-{high}"] = {
                    "count": int(mask.sum()),
                    "mean_error": round(np.mean(errors[mask]), 4),
                    "std_error": round(np.std(errors[mask]), 4)
                }
        
        return {
            "overall_mae": round(np.mean(errors), 4),
            "error_std": round(np.std(errors), 4),
            "worst_predictions": worst_predictions,
            "error_by_score_range": range_errors
        }


if __name__ == "__main__":
    # Example usage
    model_path = "../models"  # Adjust path as needed
    
    try:
        # Initialize predictor
        predictor = SentimentPredictor(model_path)
        
        # Test single prediction
        test_text = "I love this product! It's amazing and works perfectly."
        result = predictor.predict_single(test_text)
        
        print(f"Text: {test_text}")
        print(f"Prediction: {result}")
        
        # Test batch prediction
        test_texts = [
            "Great product, highly recommend!",
            "Terrible quality, waste of money.",
            "It's okay, nothing special."
        ]
        
        batch_results = predictor.predict_batch(test_texts)
        print(f"\nBatch predictions:")
        for text, result in zip(test_texts, batch_results):
            print(f"'{text}' -> {result['sentiment_score']:.2f} stars ({result['prediction_class']})")
        
        # Print model info
        model_info = predictor.get_model_info()
        print(f"\nModel info: {model_info['performance_metrics']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the model files are in the correct location.")
