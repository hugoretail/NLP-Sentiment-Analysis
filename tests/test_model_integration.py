"""
Integration tests for sentiment analysis models.
"""
import sys
import os
import unittest
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_utils import SentimentModel, ModelManager
from utils.inference import SentimentInference
from utils.text_preprocessor import TextPreprocessor

class TestModelIntegration(unittest.TestCase):
    """Test model integration and compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ModelManager()
        self.inference = SentimentInference()
        self.temp_dir = tempfile.mkdtemp()
        
    def test_model_manager_integration(self):
        """Test model manager with multiple models."""
        # Test initial state
        self.assertEqual(len(self.manager.list_models()), 0)
        self.assertIsNone(self.manager.active_model)
        
        # Test mock predictions
        result = self.manager.predict("This is a great product!")
        self.assertIn('score', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['model_name'], 'mock')
        
    def test_inference_pipeline_integration(self):
        """Test complete inference pipeline."""
        test_texts = [
            "I absolutely love this product! Amazing quality!",
            "Terrible experience, worst purchase ever.",
            "It's okay, nothing special but not bad either.",
            "Fantastic service and great value for money!",
            "Disappointing quality, not worth the price."
        ]
        
        # Test batch processing
        results = self.inference.predict_batch(test_texts)
        
        # Verify results structure
        self.assertEqual(len(results), len(test_texts))
        
        for i, result in enumerate(results):
            with self.subTest(text_index=i):
                # Check required fields
                self.assertIn('original_text', result)
                self.assertIn('score', result)
                self.assertIn('confidence', result)
                self.assertIn('sentiment_label', result)
                self.assertIn('model_type', result)
                self.assertIn('inference_time', result)
                
                # Check value ranges
                self.assertGreaterEqual(result['score'], 1.0)
                self.assertLessEqual(result['score'], 5.0)
                self.assertGreaterEqual(result['confidence'], 0.0)
                self.assertLessEqual(result['confidence'], 1.0)
                
                # Check text consistency
                self.assertEqual(result['original_text'], test_texts[i])
                
    def test_preprocessing_integration(self):
        """Test preprocessing integration with inference."""
        test_cases = [
            {
                'text': "Check this out! https://example.com @user #awesome 😍",
                'should_contain': ['check', 'awesome'],
                'should_not_contain': ['https://', '@user', '😍']
            },
            {
                'text': "I can't believe how AMAZING this is!!!",
                'should_contain': ['cannot', 'believe', 'amazing'],
                'should_not_contain': ["can't"]
            },
            {
                'text': "It's really good quality.",
                'should_contain': ['it', 'is', 'really', 'good'],
                'should_not_contain': ["it's"]
            }
        ]
        
        for case in test_cases:
            with self.subTest(text=case['text']):
                result = self.inference.predict_single(
                    case['text'], 
                    preprocess=True
                )
                
                processed_text = result['processed_text'].lower()
                
                # Check that expected words are present
                for word in case['should_contain']:
                    self.assertIn(word.lower(), processed_text)
                    
                # Check that unwanted elements are removed
                for unwanted in case['should_not_contain']:
                    self.assertNotIn(unwanted, processed_text)
                    
    def test_detailed_analysis_integration(self):
        """Test detailed analysis functionality."""
        test_text = "This product is ABSOLUTELY FANTASTIC!!! I'm so happy with my purchase! 😍🎉"
        
        result = self.inference.analyze_text_detailed(test_text)
        
        # Check detailed analysis fields
        self.assertIn('sentiment_strength', result)
        self.assertIn('text_characteristics', result)
        self.assertIn('features', result)
        
        # Check features integration
        features = result['features']
        self.assertGreater(features['exclamation_count'], 0)
        self.assertGreater(features['positive_words'], 0)
        self.assertGreater(features['capital_ratio'], 0)
        
        # Should detect characteristics
        characteristics = result['text_characteristics']
        self.assertIn('emphatic', characteristics)
        
    def test_caching_integration(self):
        """Test caching functionality."""
        # Create inference with caching enabled
        inference = SentimentInference(use_cache=True)
        
        test_text = "This is a test for caching functionality."
        
        # First prediction (not cached)
        result1 = inference.predict_single(test_text)
        self.assertFalse(result1.get('cached', True))
        
        # Second prediction (should be cached)
        result2 = inference.predict_single(test_text)
        self.assertTrue(result2.get('cached', False))
        
        # Results should be identical except for timing
        self.assertEqual(result1['score'], result2['score'])
        self.assertEqual(result1['confidence'], result2['confidence'])
        
    def test_export_functionality(self):
        """Test result export functionality."""
        test_texts = [
            "Great product!",
            "Poor quality.",
            "Average experience."
        ]
        
        results = self.inference.predict_batch(test_texts)
        
        # Test JSON export
        json_file = os.path.join(self.temp_dir, "test_results.json")
        success = self.inference.export_results(results, json_file, 'json')
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_file))
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            exported_data = json.load(f)
        self.assertEqual(len(exported_data), len(results))
        
        # Test CSV export
        csv_file = os.path.join(self.temp_dir, "test_results.csv")
        success = self.inference.export_results(results, csv_file, 'csv')
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_file))
        
    def test_statistics_integration(self):
        """Test statistics functionality."""
        # Generate some predictions to have statistics
        test_texts = [
            "Excellent product!",  # Should be positive
            "Terrible quality.",   # Should be negative
            "It's okay.",         # Should be neutral
            "Amazing experience!", # Should be positive
            "Worst ever."         # Should be negative
        ]
        
        # Create new inference instance for clean statistics
        inference = SentimentInference(use_cache=True)
        
        # Generate predictions
        for text in test_texts:
            inference.predict_single(text)
            
        # Get statistics
        stats = inference.get_statistics()
        
        # Check statistics structure
        self.assertIn('total_predictions', stats)
        self.assertIn('average_score', stats)
        self.assertIn('average_confidence', stats)
        self.assertIn('score_distribution', stats)
        
        # Check values
        self.assertEqual(stats['total_predictions'], len(test_texts))
        self.assertGreater(stats['average_score'], 0)
        self.assertGreater(stats['average_confidence'], 0)

class TestModelPerformance(unittest.TestCase):
    """Test model performance characteristics."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.inference = SentimentInference()
        
    def test_prediction_speed(self):
        """Test prediction speed for different text sizes."""
        import time
        
        test_cases = [
            "Good.",  # Very short
            "This is a good product with decent quality.",  # Medium
            " ".join(["This is a longer text with more content."] * 10)  # Long
        ]
        
        for text in test_cases:
            with self.subTest(text_length=len(text)):
                start_time = time.time()
                result = self.inference.predict_single(text)
                end_time = time.time()
                
                inference_time = end_time - start_time
                
                # Should complete within reasonable time (1 second)
                self.assertLess(inference_time, 1.0)
                
                # Result should contain timing info
                self.assertIn('inference_time', result)
                
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        import time
        
        test_texts = [f"Test text number {i}" for i in range(20)]
        
        # Time batch processing
        start_time = time.time()
        batch_results = self.inference.predict_batch(test_texts, batch_size=5)
        batch_time = time.time() - start_time
        
        # Time individual processing
        start_time = time.time()
        individual_results = []
        for text in test_texts:
            individual_results.append(self.inference.predict_single(text))
        individual_time = time.time() - start_time
        
        # Batch should not be significantly slower than individual
        # (In real implementations, batch would be faster)
        self.assertLess(batch_time, individual_time * 2)
        
        # Results should be equivalent
        self.assertEqual(len(batch_results), len(individual_results))

def run_integration_tests():
    """Run integration tests with console output."""
    print("🔗 Running Integration Tests...")
    print("=" * 50)
    
    # Test basic integration
    print("\n📊 Testing Model Integration:")
    manager = ModelManager()
    inference = SentimentInference()
    
    test_text = "This is an amazing product with excellent quality!"
    
    # Test manager
    result1 = manager.predict(test_text)
    print(f"ModelManager prediction: {result1['score']:.2f} ({result1['model_name']})")
    
    # Test inference
    result2 = inference.predict_single(test_text)
    print(f"SentimentInference prediction: {result2['score']:.2f} ({result2['sentiment_label']})")
    
    # Test detailed analysis
    detailed = inference.analyze_text_detailed(test_text)
    print(f"Detailed analysis: {detailed['sentiment_strength']} sentiment")
    print(f"Characteristics: {', '.join(detailed['text_characteristics'])}")
    
    # Test batch processing
    print("\n📦 Testing Batch Processing:")
    batch_texts = [
        "Great product!",
        "Terrible quality.",
        "Average experience.",
        "Fantastic service!",
        "Poor value for money."
    ]
    
    batch_results = inference.predict_batch(batch_texts)
    print(f"Processed {len(batch_results)} texts")
    
    # Show distribution
    positive = sum(1 for r in batch_results if r['score'] >= 3.5)
    negative = sum(1 for r in batch_results if r['score'] <= 2.5)
    neutral = len(batch_results) - positive - negative
    
    print(f"Distribution: {positive} positive, {neutral} neutral, {negative} negative")
    
    print("\n✅ Integration tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    # Run console tests
    run_integration_tests()
    
    # Run unit tests
    print("\n🔬 Running Unit Tests:")
    unittest.main(verbosity=2)
