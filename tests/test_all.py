"""
Test suite for the sentiment analysis project.
Tests data processing, model utilities, and evaluation functions.
"""

import unittest
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_processor import DataProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.evaluation import calculate_metrics, evaluate_model

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor()
        
        # Create sample data
        self.sample_data = {
            'review': [
                'This product is amazing! Love it.',
                'Terrible quality, waste of money.',
                'Good value for money, recommended.',
                'Average product, nothing special.',
                'Excellent service and fast delivery!'
            ],
            'rating': [5, 1, 4, 3, 5]
        }
        self.df = pd.DataFrame(self.sample_data)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Save sample CSV
        self.csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_data_csv(self):
        """Test loading CSV data."""
        df = self.data_processor.load_data(self.csv_path)
        self.assertEqual(len(df), 5)
        self.assertIn('review', df.columns)
        self.assertIn('rating', df.columns)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "Check out https://example.com and email me@test.com!!!"
        clean_text = self.data_processor.clean_text(dirty_text)
        
        self.assertNotIn('https://example.com', clean_text)
        self.assertNotIn('me@test.com', clean_text)
        self.assertIn('Check out', clean_text)
    
    def test_validate_ratings(self):
        """Test rating validation."""
        ratings = pd.Series([1, 2, 3, 4, 5, 6, 0, -1])
        validated = self.data_processor.validate_ratings(ratings)
        
        # Check all ratings are between 1 and 5
        self.assertTrue(all(1 <= r <= 5 for r in validated))
        self.assertEqual(validated.iloc[5], 5)  # 6 should be clipped to 5
        self.assertEqual(validated.iloc[6], 1)  # 0 should be clipped to 1
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        self.data_processor.df = self.df.copy()
        processed_df = self.data_processor.preprocess_data()
        
        self.assertEqual(len(processed_df), 5)
        self.assertTrue(self.data_processor.processed_data)
        self.assertEqual(len(self.data_processor.processed_data['texts']), 5)
    
    def test_split_data(self):
        """Test data splitting."""
        self.data_processor.df = self.df.copy()
        self.data_processor.preprocess_data()
        
        train_data, val_data, test_data = self.data_processor.split_data(
            test_size=0.2, val_size=0.2
        )
        
        total_samples = len(train_data['texts']) + len(val_data['texts']) + len(test_data['texts'])
        self.assertEqual(total_samples, 5)
    
    def test_get_data_statistics(self):
        """Test data statistics calculation."""
        self.data_processor.df = self.df.copy()
        self.data_processor.preprocess_data()
        
        stats = self.data_processor.get_data_statistics()
        
        self.assertIn('total_samples', stats)
        self.assertIn('rating_distribution', stats)
        self.assertIn('text_length_stats', stats)
        self.assertEqual(stats['total_samples'], 5)

class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_html(self):
        """Test HTML tag removal."""
        html_text = "<p>This is <b>bold</b> text.</p>"
        clean_text = self.preprocessor.clean_html(html_text)
        self.assertEqual(clean_text, "This is bold text.")
    
    def test_clean_urls(self):
        """Test URL removal."""
        url_text = "Visit https://example.com for more info"
        clean_text = self.preprocessor.clean_urls(url_text)
        self.assertEqual(clean_text, "Visit  for more info")
    
    def test_expand_contractions(self):
        """Test contraction expansion."""
        contracted_text = "I can't believe it's so good!"
        expanded_text = self.preprocessor.expand_contractions(contracted_text)
        self.assertIn("cannot", expanded_text)
        self.assertIn("it is", expanded_text)
    
    def test_preprocess_text(self):
        """Test full text preprocessing pipeline."""
        raw_text = "I can't believe this product is AMAZING!!! 😍"
        processed_text = self.preprocessor.preprocess_text(raw_text)
        
        self.assertIsInstance(processed_text, str)
        self.assertNotIn("😍", processed_text)
        self.assertIn("cannot", processed_text.lower())
    
    def test_preprocess_batch(self):
        """Test batch text preprocessing."""
        texts = [
            "This is great!",
            "I don't like it.",
            "It's okay, I guess."
        ]
        
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        self.assertEqual(len(processed_texts), 3)
        self.assertTrue(all(isinstance(text, str) for text in processed_texts))
    
    def test_get_word_frequencies(self):
        """Test word frequency calculation."""
        texts = ["great product", "product is great", "great service"]
        word_freq = self.preprocessor.get_word_frequencies(texts, top_n=5)
        
        self.assertIn("great", word_freq)
        self.assertIn("product", word_freq)
        self.assertEqual(word_freq["great"], 3)

class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample predictions
        np.random.seed(42)
        self.y_true = np.random.uniform(1, 5, 100)
        self.y_pred = self.y_true + np.random.normal(0, 0.5, 100)  # Add noise
        self.y_pred = np.clip(self.y_pred, 1, 5)  # Keep in valid range
    
    def test_calculate_metrics_regression(self):
        """Test metric calculation for regression."""
        metrics = calculate_metrics(self.y_true, self.y_pred, task_type='regression')
        
        # Check required metrics are present
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('accuracy', metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics['mae'], 0)
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['rmse'], 0)
        self.assertLessEqual(metrics['r2'], 1)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_calculate_metrics_classification(self):
        """Test metric calculation for classification."""
        # Create classification data
        y_true_class = np.random.choice([1, 2, 3, 4, 5], 100)
        y_pred_class = y_true_class.copy()
        # Add some noise
        noise_indices = np.random.choice(100, 20, replace=False)
        y_pred_class[noise_indices] = np.random.choice([1, 2, 3, 4, 5], 20)
        
        metrics = calculate_metrics(y_true_class, y_pred_class, task_type='classification')
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)
    
    def test_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = calculate_metrics(y_true, y_pred, task_type='regression')
        
        self.assertEqual(metrics['mae'], 0)
        self.assertEqual(metrics['mse'], 0)
        self.assertEqual(metrics['accuracy'], 1)
        self.assertEqual(metrics['r2'], 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample dataset
        self.sample_data = {
            'review': [
                'This product is absolutely amazing! I love everything about it.',
                'Terrible quality, complete waste of money. Very disappointed.',
                'Good value for money, would recommend to others.',
                'Average product, nothing special but does the job.',
                'Excellent service and very fast delivery! Highly recommended!',
                'Poor quality control, arrived damaged.',
                'Great features and easy to use interface.',
                'Overpriced for what you get, not worth it.',
                'Perfect for my needs, exactly what I wanted.',
                'Customer service was unhelpful and rude.'
            ] * 10,  # Repeat to have more data
            'rating': [5, 1, 4, 3, 5, 2, 4, 2, 5, 1] * 10
        }
        self.df = pd.DataFrame(self.sample_data)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'integration_test_data.csv')
        self.df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test the complete data processing and evaluation pipeline."""
        # Data processing
        data_processor = DataProcessor()
        df = data_processor.load_data(self.csv_path)
        
        self.assertEqual(len(df), 100)
        
        # Preprocess data
        processed_df = data_processor.preprocess_data()
        self.assertIsNotNone(processed_df)
        
        # Split data
        train_data, val_data, test_data = data_processor.split_data(
            test_size=0.2, val_size=0.1
        )
        
        # Check splits are reasonable
        self.assertGreater(len(train_data['texts']), 0)
        self.assertGreater(len(test_data['texts']), 0)
        
        # Text preprocessing
        preprocessor = TextPreprocessor()
        processed_texts = preprocessor.preprocess_batch(test_data['texts'])
        
        self.assertEqual(len(processed_texts), len(test_data['texts']))
        
        # Create dummy predictions for evaluation
        y_true = np.array(test_data['ratings'])
        y_pred = y_true + np.random.normal(0, 0.3, len(y_true))
        y_pred = np.clip(y_pred, 1, 5)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, task_type='regression')
        
        # Verify metrics are calculated correctly
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('accuracy', metrics)
        
        # Get data statistics
        stats = data_processor.get_data_statistics()
        self.assertIn('total_samples', stats)
        self.assertEqual(stats['total_samples'], 100)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)
