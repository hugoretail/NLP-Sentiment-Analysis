"""
Inference utilities for sentiment analysis.
"""
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import hashlib
import pickle
from .model_utils import SentimentModel, ModelManager
from .text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class SentimentInference:
    """Complete inference pipeline for sentiment analysis."""
    
    def __init__(self, model_path: Optional[str] = None, use_cache: bool = True):
        self.model_manager = ModelManager()
        self.preprocessor = TextPreprocessor()
        self.use_cache = use_cache
        self.cache = {}
        self.cache_file = "inference_cache.pkl"
        self.load_cache()
        
        # Load default model if provided
        if model_path:
            self.model_manager.load_model("default", model_path)
            
    def load_cache(self):
        """Load inference cache from disk."""
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.cache = {}
                
    def save_cache(self):
        """Save inference cache to disk."""
        if self.use_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")
                
    def _get_cache_key(self, text: str, preprocess_options: Dict[str, Any] = None) -> str:
        """Generate cache key for text and options."""
        cache_data = {
            'text': text,
            'options': preprocess_options or {}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
        
    def predict_single(self, 
                      text: str, 
                      preprocess: bool = True,
                      preprocess_options: Dict[str, bool] = None,
                      return_features: bool = False,
                      use_cache: bool = None) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            preprocess: Whether to preprocess the text
            preprocess_options: Options for text preprocessing
            return_features: Whether to return text features
            use_cache: Whether to use caching (overrides instance setting)
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Check cache
        should_use_cache = use_cache if use_cache is not None else self.use_cache
        cache_key = None
        
        if should_use_cache:
            cache_key = self._get_cache_key(text, preprocess_options)
            if cache_key in self.cache:
                result = self.cache[cache_key].copy()
                result['cached'] = True
                result['inference_time'] = time.time() - start_time
                return result
        
        # Preprocess text if requested
        original_text = text
        if preprocess:
            text = self.preprocessor.preprocess_text(text, preprocess_options)
            
        # Get prediction from model
        prediction = self.model_manager.predict(text)
        
        # Extract features if requested
        features = {}
        if return_features:
            features = self.preprocessor.extract_features(original_text)
            
        # Compile result
        result = {
            'original_text': original_text,
            'processed_text': text if preprocess else None,
            'score': prediction['score'],
            'confidence': prediction['confidence'],
            'model_type': prediction.get('model_type', 'unknown'),
            'model_name': prediction.get('model_name', 'default'),
            'sentiment_label': self._score_to_label(prediction['score']),
            'inference_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'cached': False
        }
        
        if return_features:
            result['features'] = features
            
        # Cache result
        if should_use_cache and cache_key:
            self.cache[cache_key] = result.copy()
            # Limit cache size
            if len(self.cache) > 10000:
                # Remove oldest entries
                sorted_cache = sorted(self.cache.items(), 
                                    key=lambda x: x[1].get('timestamp', ''))
                self.cache = dict(sorted_cache[-5000:])
                
        return result
        
    def predict_batch(self, 
                     texts: List[str],
                     preprocess: bool = True,
                     preprocess_options: Dict[str, bool] = None,
                     return_features: bool = False,
                     batch_size: int = 32,
                     show_progress: bool = False) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            preprocess: Whether to preprocess the texts
            preprocess_options: Options for text preprocessing
            return_features: Whether to return text features
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            List of prediction results
        """
        results = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                print(f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")
                
            batch_results = []
            for text in batch_texts:
                result = self.predict_single(
                    text, 
                    preprocess=preprocess,
                    preprocess_options=preprocess_options,
                    return_features=return_features
                )
                batch_results.append(result)
                
            results.extend(batch_results)
            
        # Save cache after batch processing
        if self.use_cache:
            self.save_cache()
            
        return results
        
    def _score_to_label(self, score: float) -> str:
        """Convert numerical score to sentiment label."""
        if score >= 4.5:
            return "very positive"
        elif score >= 3.5:
            return "positive"
        elif score >= 2.5:
            return "neutral"
        elif score >= 1.5:
            return "negative"
        else:
            return "very negative"
            
    def analyze_text_detailed(self, text: str) -> Dict[str, Any]:
        """Perform detailed analysis of a text."""
        # Get basic prediction
        result = self.predict_single(text, return_features=True)
        
        # Add detailed analysis
        features = result.get('features', {})
        
        # Sentiment strength
        score = result['score']
        if score >= 4.0:
            sentiment_strength = "strong"
        elif score >= 3.5 or score <= 1.5:
            sentiment_strength = "moderate"
        else:
            sentiment_strength = "weak"
            
        # Text characteristics
        characteristics = []
        if features.get('exclamation_count', 0) > 2:
            characteristics.append("emphatic")
        if features.get('capital_ratio', 0) > 0.3:
            characteristics.append("shouty")
        if features.get('word_count', 0) > 50:
            characteristics.append("detailed")
        elif features.get('word_count', 0) < 10:
            characteristics.append("brief")
            
        result['sentiment_strength'] = sentiment_strength
        result['text_characteristics'] = characteristics
        
        return result
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'loaded_models': self.model_manager.list_models(),
            'active_model': self.model_manager.active_model,
            'model_info': self.model_manager.get_model_info(),
            'cache_size': len(self.cache) if self.use_cache else 0
        }
        
    def load_model(self, name: str, model_path: str) -> bool:
        """Load a new model."""
        return self.model_manager.load_model(name, model_path)
        
    def set_active_model(self, name: str) -> bool:
        """Set the active model for predictions."""
        return self.model_manager.set_active_model(name)
        
    def export_results(self, results: List[Dict[str, Any]], 
                      filename: str, format: str = 'json') -> bool:
        """
        Export prediction results to file.
        
        Args:
            results: List of prediction results
            filename: Output filename
            format: Export format ('json', 'csv')
            
        Returns:
            Success status
        """
        try:
            if format.lower() == 'json':
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'csv':
                import csv
                if not results:
                    return False
                    
                fieldnames = results[0].keys()
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in results:
                        # Flatten nested dictionaries for CSV
                        flat_result = {}
                        for key, value in result.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    flat_result[f"{key}_{subkey}"] = subvalue
                            else:
                                flat_result[key] = value
                        writer.writerow(flat_result)
                        
            logger.info(f"Exported {len(results)} results to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
            
    def clear_cache(self):
        """Clear the inference cache."""
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception as e:
                logger.warning(f"Could not remove cache file: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        if not self.cache:
            return {"message": "No cached predictions available"}
            
        scores = [result['score'] for result in self.cache.values()]
        confidences = [result['confidence'] for result in self.cache.values()]
        
        return {
            'total_predictions': len(self.cache),
            'average_score': sum(scores) / len(scores),
            'average_confidence': sum(confidences) / len(confidences),
            'score_distribution': {
                'very_positive': len([s for s in scores if s >= 4.5]),
                'positive': len([s for s in scores if 3.5 <= s < 4.5]),
                'neutral': len([s for s in scores if 2.5 <= s < 3.5]),
                'negative': len([s for s in scores if 1.5 <= s < 2.5]),
                'very_negative': len([s for s in scores if s < 1.5])
            },
            'model_usage': {
                result['model_name']: len([r for r in self.cache.values() 
                                         if r.get('model_name') == result['model_name']])
                for result in self.cache.values()
            }
        }
