"""
Inference module for sentiment analysis with optimized performance.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import logging

from src.model_utils import SentimentPredictor
from src.data_preprocessing import TwitterTextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentInferenceEngine:
    """
    High-level inference engine with caching and performance optimization.
    """
    
    def __init__(self, model_path: str, enable_cache: bool = True, cache_size: int = 1000):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model
            enable_cache: Whether to enable result caching
            cache_size: Maximum number of cached results
        """
        self.model_path = model_path
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # Initialize components
        self.preprocessor = TwitterTextPreprocessor(remove_stopwords=False, lowercase=True)
        self.predictor = SentimentPredictor(model_path)
        
        # Initialize cache
        self._cache = {} if enable_cache else None
        self._cache_access_times = {} if enable_cache else None
        self._cache_lock = threading.Lock() if enable_cache else None
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.cache_hits = 0
        
        logger.info("Sentiment inference engine initialized")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key string
        """
        # Use hash of preprocessed text as cache key
        preprocessed = self.preprocessor.preprocess_text(text)
        return str(hash(preprocessed))
    
    def _update_cache(self, cache_key: str, result: Dict[str, Any]):
        """
        Update cache with new result.
        
        Args:
            cache_key: Cache key
            result: Prediction result
        """
        if not self.enable_cache:
            return
        
        with self._cache_lock:
            # If cache is full, remove oldest entry
            if len(self._cache) >= self.cache_size:
                oldest_key = min(self._cache_access_times.keys(), 
                               key=lambda k: self._cache_access_times[k])
                del self._cache[oldest_key]
                del self._cache_access_times[oldest_key]
            
            # Add new result
            self._cache[cache_key] = result
            self._cache_access_times[cache_key] = time.time()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get result from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        if not self.enable_cache:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                # Update access time
                self._cache_access_times[cache_key] = time.time()
                self.cache_hits += 1
                return self._cache[cache_key].copy()
        
        return None
    
    def predict_sentiment(self, text: str, include_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Predict sentiment for a single text with caching and preprocessing.
        
        Args:
            text: Input text
            include_preprocessing: Whether to apply text preprocessing
            
        Returns:
            Dictionary with prediction results and metadata
        """
        start_time = time.time()
        
        # Input validation
        if not text or not isinstance(text, str):
            return {
                "error": "Invalid input text",
                "sentiment_score": 3.0,
                "confidence": 0.0,
                "prediction_class": "neutral",
                "processing_time": 0.0,
                "cached": False
            }
        
        # Check cache
        cache_key = self._get_cache_key(text)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            cached_result.update({
                "processing_time": time.time() - start_time,
                "cached": True
            })
            return cached_result
        
        try:
            # Preprocess text if requested
            processed_text = text
            if include_preprocessing:
                processed_text = self.preprocessor.preprocess_text(text)
            
            # Make prediction
            prediction = self.predictor.predict_single(processed_text)
            
            # Add metadata
            result = {
                **prediction,
                "original_text": text,
                "processed_text": processed_text if include_preprocessing else None,
                "processing_time": time.time() - start_time,
                "cached": False,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update cache
            self._update_cache(cache_key, result)
            
            # Update performance tracking
            self.inference_count += 1
            self.total_inference_time += result["processing_time"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return {
                "error": str(e),
                "sentiment_score": 3.0,
                "confidence": 0.0,
                "prediction_class": "neutral",
                "processing_time": time.time() - start_time,
                "cached": False
            }
    
    def predict_batch(self, texts: List[str], batch_size: int = 16, 
                     include_preprocessing: bool = True) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts with optimization.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            include_preprocessing: Whether to apply text preprocessing
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        results = []
        
        # Separate cached and non-cached texts
        cached_results = {}
        non_cached_texts = []
        non_cached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                cached_results[i] = cached_result
            else:
                non_cached_texts.append(text)
                non_cached_indices.append(i)
        
        # Process non-cached texts
        if non_cached_texts:
            # Preprocess if requested
            processed_texts = non_cached_texts
            if include_preprocessing:
                processed_texts = self.preprocessor.preprocess_batch(non_cached_texts)
            
            # Get predictions
            predictions = self.predictor.predict_batch(processed_texts, batch_size)
            
            # Add metadata and cache results
            for i, (original_idx, original_text, processed_text, prediction) in enumerate(
                zip(non_cached_indices, non_cached_texts, processed_texts, predictions)
            ):
                result = {
                    **prediction,
                    "original_text": original_text,
                    "processed_text": processed_text if include_preprocessing else None,
                    "cached": False,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache result
                cache_key = self._get_cache_key(original_text)
                self._update_cache(cache_key, result)
                
                cached_results[original_idx] = result
        
        # Combine results in original order
        for i in range(len(texts)):
            result = cached_results[i]
            result["processing_time"] = time.time() - start_time  # Approximate for batch
            results.append(result)
        
        # Update performance tracking
        self.inference_count += len(texts)
        self.total_inference_time += time.time() - start_time
        
        return results
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis with detailed insights.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detailed analysis
        """
        # Get basic prediction
        prediction = self.predict_sentiment(text)
        
        if "error" in prediction:
            return prediction
        
        # Extract insights
        sentiment_score = prediction["sentiment_score"]
        confidence = prediction["confidence"]
        prediction_class = prediction["prediction_class"]
        
        # Determine sentiment strength
        if prediction_class == "positive":
            if sentiment_score >= 4.5:
                strength = "very positive"
            elif sentiment_score >= 4.0:
                strength = "positive"
            else:
                strength = "somewhat positive"
        elif prediction_class == "negative":
            if sentiment_score <= 1.5:
                strength = "very negative"
            elif sentiment_score <= 2.0:
                strength = "negative"
            else:
                strength = "somewhat negative"
        else:
            strength = "neutral"
        
        # Calculate star rating display
        stars_full = int(sentiment_score)
        stars_half = 1 if (sentiment_score - stars_full) >= 0.5 else 0
        stars_empty = 5 - stars_full - stars_half
        
        # Confidence level description
        if confidence >= 0.8:
            confidence_level = "very high"
        elif confidence >= 0.6:
            confidence_level = "high"
        elif confidence >= 0.4:
            confidence_level = "moderate"
        elif confidence >= 0.2:
            confidence_level = "low"
        else:
            confidence_level = "very low"
        
        # Enhanced result
        enhanced_result = {
            **prediction,
            "sentiment_analysis": {
                "sentiment_strength": strength,
                "confidence_level": confidence_level,
                "star_rating": {
                    "full_stars": stars_full,
                    "half_stars": stars_half,
                    "empty_stars": stars_empty,
                    "rating_text": f"{sentiment_score:.1f}/5.0"
                },
                "interpretation": self._generate_interpretation(sentiment_score, confidence, strength)
            }
        }
        
        return enhanced_result
    
    def _generate_interpretation(self, score: float, confidence: float, strength: str) -> str:
        """
        Generate human-readable interpretation of the sentiment analysis.
        
        Args:
            score: Sentiment score
            confidence: Confidence level
            strength: Sentiment strength description
            
        Returns:
            Human-readable interpretation
        """
        base_interpretation = f"This text expresses {strength} sentiment"
        
        if confidence >= 0.7:
            confidence_text = "with high confidence"
        elif confidence >= 0.4:
            confidence_text = "with moderate confidence"
        else:
            confidence_text = "with low confidence"
        
        score_context = f"(scored {score:.1f} out of 5.0)"
        
        return f"{base_interpretation} {confidence_text} {score_context}."
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the inference engine.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 else 0.0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.inference_count 
            if self.inference_count > 0 else 0.0
        )
        
        return {
            "total_inferences": self.inference_count,
            "total_inference_time": round(self.total_inference_time, 4),
            "average_inference_time": round(avg_inference_time, 4),
            "cache_enabled": self.enable_cache,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "cache_size": len(self._cache) if self.enable_cache else 0,
            "max_cache_size": self.cache_size,
            "model_info": self.predictor.get_model_info()
        }
    
    def clear_cache(self):
        """Clear the inference cache."""
        if self.enable_cache:
            with self._cache_lock:
                self._cache.clear()
                self._cache_access_times.clear()
                self.cache_hits = 0
            logger.info("Inference cache cleared")
    
    def export_cache(self, filepath: str):
        """
        Export cache to file for persistence.
        
        Args:
            filepath: Path to export file
        """
        if not self.enable_cache:
            logger.warning("Cache is disabled, nothing to export")
            return
        
        try:
            with self._cache_lock:
                cache_data = {
                    "cache": self._cache,
                    "access_times": self._cache_access_times,
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "cache_size": len(self._cache),
                        "model_path": str(self.model_path)
                    }
                }
            
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cache exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting cache: {e}")
    
    def import_cache(self, filepath: str):
        """
        Import cache from file.
        
        Args:
            filepath: Path to import file
        """
        if not self.enable_cache:
            logger.warning("Cache is disabled, cannot import")
            return
        
        try:
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
            
            with self._cache_lock:
                self._cache = cache_data.get("cache", {})
                self._cache_access_times = cache_data.get("access_times", {})
            
            logger.info(f"Cache imported from {filepath} with {len(self._cache)} entries")
            
        except Exception as e:
            logger.error(f"Error importing cache: {e}")


# Global inference engine instance
_inference_engine = None

def get_inference_engine(model_path: str = None) -> SentimentInferenceEngine:
    """
    Get singleton inference engine instance.
    
    Args:
        model_path: Path to model (required for first call)
        
    Returns:
        SentimentInferenceEngine instance
    """
    global _inference_engine
    
    if _inference_engine is None:
        if model_path is None:
            raise ValueError("model_path required for first initialization")
        _inference_engine = SentimentInferenceEngine(model_path)
    
    return _inference_engine


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize inference engine
        engine = SentimentInferenceEngine("../models")
        
        # Test single prediction
        test_text = "I absolutely love this product! It exceeded all my expectations."
        result = engine.analyze_text_sentiment(test_text)
        
        print(f"Text: {test_text}")
        print(f"Sentiment: {result['sentiment_analysis']['interpretation']}")
        print(f"Rating: {result['sentiment_analysis']['star_rating']['rating_text']}")
        print(f"Confidence: {result['sentiment_analysis']['confidence_level']}")
        
        # Test batch prediction
        test_texts = [
            "Great product, highly recommend!",
            "Terrible quality, complete waste of money.",
            "It's okay, nothing special but works fine.",
            "Amazing! Best purchase I've made in years!",
            "Poor customer service and late delivery."
        ]
        
        batch_results = engine.predict_batch(test_texts)
        
        print(f"\nBatch Results:")
        for text, result in zip(test_texts, batch_results):
            print(f"'{text[:30]}...' -> {result['sentiment_score']:.1f}/5.0 ({result['prediction_class']})")
        
        # Performance stats
        stats = engine.get_performance_stats()
        print(f"\nPerformance: {stats['total_inferences']} inferences, "
              f"{stats['average_inference_time']*1000:.1f}ms avg, "
              f"{stats['cache_hit_rate']*100:.1f}% cache hit rate")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the model files are in the correct location.")
