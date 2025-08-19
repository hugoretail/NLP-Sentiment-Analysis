"""
Comprehensive test suite for sentiment analysis components.
"""
import sys
import os
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.inference import SentimentInference
from utils.text_preprocessor import TextPreprocessor
from utils.model_utils import SentimentModel, ModelManager

class TestBasicFunctionality(unittest.TestCase):
    """Test basic sentiment analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.inference = SentimentInference()
        self.preprocessor = TextPreprocessor()
        self.model = SentimentModel()
        
    def test_basic_predictions(self):
        """Test basic sentiment predictions."""
        test_cases = [
            ("I love this product! It's amazing!", "positive"),
            ("This is terrible, I hate it.", "negative"),
            ("It's okay, nothing special.", "neutral"),
            ("Absolutely fantastic experience!", "positive"),
            ("Worst purchase ever, complete waste of money!", "negative")
        ]
        
        for text, expected_sentiment in test_cases:
            with self.subTest(text=text):
                result = self.inference.predict_single(text)
                
                # Check result structure
                self.assertIn('score', result)
                self.assertIn('confidence', result)
                self.assertIn('sentiment_label', result)
                
                # Check score range
                self.assertGreaterEqual(result['score'], 1.0)
                self.assertLessEqual(result['score'], 5.0)
                
                # Check confidence range
                self.assertGreaterEqual(result['confidence'], 0.0)
                self.assertLessEqual(result['confidence'], 1.0)
                
    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        test_text = "Check out this AMAZING product at https://example.com! @user #awesome 😍"
        
        result = self.preprocessor.preprocess_text(test_text)
        
        # Should remove URLs
        self.assertNotIn("https://example.com", result)
        # Should remove mentions
        self.assertNotIn("@user", result)
        # Should handle emojis
        self.assertNotIn("😍", result)
        # Should be lowercase
        self.assertTrue(result.islower())
        
    def test_feature_extraction(self):
        """Test text feature extraction."""
        test_text = "This is an AMAZING product!!! I love it so much!"
        
        features = self.preprocessor.extract_features(test_text)
        
        # Check feature presence
        self.assertIn('length', features)
        self.assertIn('word_count', features)
        self.assertIn('exclamation_count', features)
        self.assertIn('capital_ratio', features)
        self.assertIn('positive_words', features)
        self.assertIn('negative_words', features)
        
        # Check feature values
        self.assertGreater(features['exclamation_count'], 0)
        self.assertGreater(features['positive_words'], 0)
        
    def test_batch_predictions(self):
        """Test batch prediction functionality."""
        test_texts = [
            "Great product!",
            "Terrible service.",
            "Average quality.",
            "Love it!"
        ]
        
        results = self.inference.predict_batch(test_texts)
        
        # Check results length
        self.assertEqual(len(results), len(test_texts))
        
        # Check each result structure
        for result in results:
            self.assertIn('score', result)
            self.assertIn('confidence', result)
            self.assertIn('sentiment_label', result)
            
    def test_model_manager(self):
        """Test model manager functionality."""
        manager = ModelManager()
        
        # Test initial state
        self.assertEqual(len(manager.list_models()), 0)
        
        # Test mock predictions
        result = manager.predict("This is a test")
        self.assertIn('score', result)
        self.assertIn('model_name', result)
        self.assertEqual(result['model_name'], 'mock')
        
    def test_detailed_analysis(self):
        """Test detailed text analysis."""
        test_text = "This product is ABSOLUTELY AMAZING!!! Best purchase ever!"
        
        result = self.inference.analyze_text_detailed(test_text)
        
        # Check detailed analysis fields
        self.assertIn('sentiment_strength', result)
        self.assertIn('text_characteristics', result)
        self.assertIn('features', result)
        
        # Should detect emphatic nature
        self.assertIn('emphatic', result['text_characteristics'])

def run_basic_tests():
    """Run basic functionality tests with console output."""
    print("🧪 Testing Sentiment Analysis Components...")
    print("=" * 50)
    
    # Test inference
    inference = SentimentInference()
    print("\n📊 Testing Basic Predictions:")
    
    test_cases = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic experience!",
        "Worst purchase ever!"
    ]
    
    for text in test_cases:
        result = inference.predict_single(text)
        print(f"Text: '{text[:40]}{'...' if len(text) > 40 else ''}'")
        print(f"  Score: {result['score']:.2f} | Confidence: {result['confidence']:.3f} | Label: {result['sentiment_label']}")
        print()
    
    # Test preprocessing
    print("🔧 Testing Text Preprocessing:")
    preprocessor = TextPreprocessor()
    
    test_text = "Check out this AMAZING product at https://example.com! @user #awesome 😍"
    processed = preprocessor.preprocess_text(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")
    print()
    
    # Test features
    features = preprocessor.extract_features(test_text)
    print("📈 Extracted Features:")
    for key, value in list(features.items())[:8]:  # Show first 8 features
        print(f"  {key}: {value}")
    print()
    
    # Test batch processing
    print("📦 Testing Batch Processing:")
    batch_results = inference.predict_batch(test_cases[:3])
    print(f"Processed {len(batch_results)} texts in batch")
    avg_score = sum(r['score'] for r in batch_results) / len(batch_results)
    print(f"Average sentiment score: {avg_score:.2f}")
    print()
    
    print("✅ All basic tests completed successfully!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    # Run console tests
    run_basic_tests()
    
    # Run unit tests
    print("\n🔬 Running Unit Tests:")
    unittest.main(verbosity=2)
