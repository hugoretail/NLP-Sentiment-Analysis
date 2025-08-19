# -*- coding: utf-8 -*-
"""Inference utilities for sentiment analysis."""
from .model_utils import SentimentModel
from .text_preprocessor import TextPreprocessor
import logging

class SentimentInference:
    """Complete inference pipeline for sentiment analysis."""
ECHO is on.
    def __init__(self, model_path=None):
        self.model = SentimentModel(model_path)
        self.preprocessor = TextPreprocessor()
        self.model_loaded = self.model.load_model() if model_path else False
ECHO is on.
    def predict_single(self, text: str) -> dict:
        """Predict sentiment for a single text."""
        cleaned_text = self.preprocessor.clean_text(text)
ECHO is on.
        if not self.model_loaded:
            # Mock prediction for demo
            import random
            score = round(random.uniform(1.5, 4.5), 2)
            confidence = round(random.uniform(0.7, 0.95), 3)
            return {
                "text": text,
                "cleaned_text": cleaned_text,
                "score": score,
                "confidence": confidence,
                "stars": int(round(score)),
                "sentiment": "positive" if score > 3 else "negative" if score < 3 else "neutral"
            }
ECHO is on.
        result = self.model.predict(cleaned_text)
        result["text"] = text
        result["cleaned_text"] = cleaned_text
        result["stars"] = int(round(result["score"]))
        result["sentiment"] = "positive" if result["score"] > 3 else "negative" if result["score"] < 3 else "neutral"
ECHO is on.
        return result
ECHO is on.
    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for multiple texts."""
        return [self.predict_single(text) for text in texts]
