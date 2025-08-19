"""
"""
"""
Sentiment Analysis Utilities Package

This package provides comprehensive utilities for sentiment analysis including:
- Model management and inference
- Text preprocessing and feature extraction  
- Complete inference pipeline with caching
- Web interface components

Classes:
    SentimentModel: Model wrapper for loading and using sentiment models
    ModelManager: Manager for multiple sentiment models
    TextPreprocessor: Comprehensive text preprocessing utilities
    SentimentInference: Complete inference pipeline
"""

from .model_utils import SentimentModel, ModelManager
from .text_preprocessor import TextPreprocessor
from .inference import SentimentInference

__all__ = [
    'SentimentModel',
    'ModelManager', 
    'TextPreprocessor',
    'SentimentInference'
]

__version__ = '1.0.0'
"""

from .model_utils import SentimentModel, ModelManager
from .inference import SentimentInference
from .text_preprocessor import TextPreprocessor

__version__ = "1.0.0"
__author__ = "Hugo Retail"

__all__ = [
    "SentimentModel",
    "SentimentInference", 
    "TextPreprocessor"
] for sentiment analysis project.
Contains data processing, model utilities, and helper functions.
"""

from .data_processor import DataProcessor
from .model_utils import ModelUtils, load_model, save_model
from .text_preprocessor import TextPreprocessor
from .evaluation import evaluate_model, calculate_metrics

__all__ = [
    'DataProcessor',
    'ModelUtils', 
    'load_model',
    'save_model',
    'TextPreprocessor',
    'evaluate_model',
    'calculate_metrics'
]
