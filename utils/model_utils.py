# -*- coding: utf-8 -*-
"""Model utilities for sentiment analysis."""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import logging

class SentimentModel:
    """Sentiment analysis model wrapper."""
ECHO is on.
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ECHO is on.
    def load_model(self):
        """Load the sentiment model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
ECHO is on.
    def predict(self, text):
        """Predict sentiment for a single text."""
        if not self.model:
            return {"score": 3.0, "confidence": 0.5}
ECHO is on.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
ECHO is on.
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            score = torch.clamp(logits.squeeze(), 0, 1).item() * 4 + 1
            confidence = torch.sigmoid(logits).item()
ECHO is on.
        return {"score": round(score, 2), "confidence": round(confidence, 3)}
