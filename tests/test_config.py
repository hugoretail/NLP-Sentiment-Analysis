"""
Test configuration and utilities for sentiment analysis tests.
"""
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any

# Test configuration settings
TEST_CONFIG = {
    'data': {
        'sample_size': 100,
        'test_size': 0.2,
        'validation_size': 0.1,
        'random_state': 42,
        'min_text_length': 5,
        'max_text_length': 500
    },
    'models': {
        'mock': {
            'enabled': True,
            'score_range': [1.0, 5.0],
            'confidence_range': [0.5, 1.0]
        },
        'sklearn': {
            'test_models': ['random_forest', 'ridge', 'svr'],
            'enabled': False  # Disabled for mock testing
        },
        'transformers': {
            'test_models': ['distilbert-base-uncased', 'roberta-base'],
            'enabled': False  # Disabled for mock testing
        }
    },
    'preprocessing': {
        'test_all_options': True,
        'test_combinations': [
            {'lowercase': True, 'remove_urls': True, 'expand_contractions': True},
            {'lowercase': False, 'remove_punctuation': True, 'handle_emojis': True},
            {'remove_special_chars': True, 'normalize_unicode': True}
        ]
    },
    'performance': {
        'max_inference_time': 1.0,  # seconds
        'batch_size': 32,
        'cache_enabled': True
    },
    'output': {
        'export_formats': ['json', 'csv'],
        'include_features': True,
        'detailed_analysis': True
    }
}

# Test data samples
SAMPLE_TEXTS = {
    'positive': [
        "I absolutely love this product! Amazing quality and great value.",
        "Fantastic experience! Highly recommend to everyone.",
        "Best purchase I've made in years. Excellent service!",
        "Outstanding quality and fast delivery. Very satisfied!",
        "Perfect product, exactly what I was looking for. 5 stars!",
        "Incredible value for money. Will definitely buy again!",
        "Amazing customer service and beautiful product design.",
        "Exceeded all my expectations. Truly exceptional!",
        "Wonderful experience from start to finish. Love it!",
        "Top-notch quality and attention to detail. Brilliant!"
    ],
    'negative': [
        "Terrible product, complete waste of money. Very disappointed.",
        "Worst purchase ever. Poor quality and bad customer service.",
        "Absolutely horrible experience. Would not recommend to anyone.",
        "Disappointing quality, not worth the price at all.",
        "Broken on arrival and terrible replacement service.",
        "Useless product that doesn't work as advertised.",
        "Poor quality materials and sloppy construction.",
        "Frustrated with this purchase. Total waste of time and money.",
        "Defective product with no proper customer support.",
        "Cheap quality and overpriced. Very unsatisfied."
    ],
    'neutral': [
        "It's an okay product, nothing special but does the job.",
        "Average quality for the price. Neither good nor bad.",
        "Decent product but could be better in some areas.",
        "Standard quality, meets basic expectations.",
        "It works as expected, no major complaints or praise.",
        "Satisfactory product with room for improvement.",
        "Fair quality for the price point. Could be worse.",
        "Acceptable performance, meets minimum requirements.",
        "Middle of the road product. Gets the job done.",
        "Adequate quality, nothing to write home about."
    ],
    'mixed': [
        "Great design but poor customer service experience.",
        "Good quality product but expensive for what you get.",
        "Love the features but hate the complicated setup process.",
        "Excellent build quality but disappointing battery life.",
        "Beautiful appearance but functionality could be better.",
        "Fast delivery but product quality was not as expected.",
        "Good value but packaging could be more secure.",
        "Nice features but difficult user interface design.",
        "Quality materials but assembly instructions unclear.",
        "Works well but customer support is unresponsive."
    ]
}

# Test edge cases
EDGE_CASE_TEXTS = [
    "",  # Empty string
    "   ",  # Whitespace only
    "a",  # Single character
    "OK",  # Very short
    "!" * 100,  # Repeated punctuation
    "😀😀😀😀😀",  # Only emojis
    "https://example.com",  # Only URL
    "@user #hashtag",  # Only mentions and hashtags
    "SHOUTING TEXT IN ALL CAPS!!!",  # All caps
    "Can't won't shouldn't wouldn't",  # Many contractions
    "Check out https://example.com @user #amazing 😍🎉👏",  # Mixed content
    "This is a very " + "long " * 50 + "text to test processing of lengthy content.",  # Very long
]

# Performance test data
PERFORMANCE_TEST_TEXTS = [
    f"Performance test text number {i} with varying content and length."
    for i in range(100)
]

class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def get_sample_texts(category: str = None, count: int = None) -> List[str]:
        """
        Get sample texts for testing.
        
        Args:
            category: Text category ('positive', 'negative', 'neutral', 'mixed')
            count: Number of texts to return
            
        Returns:
            List of sample texts
        """
        if category and category in SAMPLE_TEXTS:
            texts = SAMPLE_TEXTS[category]
        else:
            # Return all categories mixed
            texts = []
            for cat_texts in SAMPLE_TEXTS.values():
                texts.extend(cat_texts)
                
        if count:
            return texts[:count]
        return texts
    
    @staticmethod
    def get_edge_cases() -> List[str]:
        """Get edge case texts for testing."""
        return EDGE_CASE_TEXTS.copy()
    
    @staticmethod
    def get_performance_texts(count: int = 100) -> List[str]:
        """Get texts for performance testing."""
        return PERFORMANCE_TEST_TEXTS[:count]
    
    @staticmethod
    def create_labeled_dataset(size: int = 100) -> List[Dict[str, Any]]:
        """
        Create a labeled dataset for testing.
        
        Args:
            size: Size of the dataset
            
        Returns:
            List of dictionaries with 'text' and 'label' keys
        """
        dataset = []
        categories = list(SAMPLE_TEXTS.keys())
        
        texts_per_category = size // len(categories)
        
        for category in categories:
            texts = SAMPLE_TEXTS[category]
            
            for i in range(min(texts_per_category, len(texts))):
                if category == 'positive':
                    label = 4.5  # High positive score
                elif category == 'negative':
                    label = 1.5  # Low negative score
                elif category == 'neutral':
                    label = 3.0  # Neutral score
                else:  # mixed
                    label = 3.2  # Slightly positive
                    
                dataset.append({
                    'text': texts[i],
                    'label': label,
                    'category': category
                })
                
        return dataset

class TestFileManager:
    """Utility for managing test files and directories."""
    
    def __init__(self):
        self.temp_dir = None
        
    def setup_temp_directory(self) -> str:
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp(prefix='sentiment_test_')
        return self.temp_dir
        
    def cleanup_temp_directory(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
    def create_test_data_file(self, filename: str, data: Any, format: str = 'json') -> str:
        """
        Create a test data file.
        
        Args:
            filename: Name of the file
            data: Data to write
            format: File format ('json', 'csv', 'txt')
            
        Returns:
            Path to created file
        """
        if not self.temp_dir:
            self.setup_temp_directory()
            
        filepath = os.path.join(self.temp_dir, filename)
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if data and isinstance(data[0], dict):
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        elif format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, list):
                    f.write('\n'.join(str(item) for item in data))
                else:
                    f.write(str(data))
                    
        return filepath

class TestAssertions:
    """Custom assertions for sentiment analysis testing."""
    
    @staticmethod
    def assert_valid_prediction(result: Dict[str, Any], testcase):
        """Assert that a prediction result is valid."""
        # Check required fields
        required_fields = ['score', 'confidence', 'sentiment_label']
        for field in required_fields:
            testcase.assertIn(field, result, f"Missing required field: {field}")
            
        # Check score range
        testcase.assertGreaterEqual(result['score'], 1.0, "Score below minimum")
        testcase.assertLessEqual(result['score'], 5.0, "Score above maximum")
        
        # Check confidence range
        testcase.assertGreaterEqual(result['confidence'], 0.0, "Confidence below minimum")
        testcase.assertLessEqual(result['confidence'], 1.0, "Confidence above maximum")
        
        # Check sentiment label
        valid_labels = ['very positive', 'positive', 'neutral', 'negative', 'very negative']
        testcase.assertIn(result['sentiment_label'], valid_labels, "Invalid sentiment label")
        
    @staticmethod
    def assert_consistent_predictions(results: List[Dict[str, Any]], testcase):
        """Assert that batch predictions are consistent."""
        testcase.assertGreater(len(results), 0, "No results to check")
        
        for i, result in enumerate(results):
            with testcase.subTest(result_index=i):
                TestAssertions.assert_valid_prediction(result, testcase)
                
    @staticmethod
    def assert_performance_acceptable(inference_time: float, max_time: float, testcase):
        """Assert that performance is within acceptable limits."""
        testcase.assertLess(inference_time, max_time, 
                           f"Inference took {inference_time:.3f}s, max allowed: {max_time}s")

# Test utilities
def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def setup_test_environment():
    """Set up the test environment."""
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def get_test_config(key: str = None) -> Any:
    """
    Get test configuration.
    
    Args:
        key: Configuration key (dot notation supported, e.g., 'data.sample_size')
        
    Returns:
        Configuration value or entire config if key is None
    """
    if key is None:
        return TEST_CONFIG
        
    keys = key.split('.')
    config = TEST_CONFIG
    
    for k in keys:
        if k in config:
            config = config[k]
        else:
            raise KeyError(f"Configuration key not found: {key}")
            
    return config

# Initialize test environment when module is imported
setup_test_environment()

def get_test_data_path():
    """Get the test data directory path."""
    return get_project_root() / TEST_CONFIG['paths']['test_data_dir']

def get_test_models_path():
    """Get the test models directory path."""
    return get_project_root() / TEST_CONFIG['paths']['test_models_dir']

def get_test_results_path():
    """Get the test results directory path."""
    return get_project_root() / TEST_CONFIG['paths']['test_results_dir']

def create_sample_dataset(size=None):
    """Create a sample dataset for testing."""
    import pandas as pd
    import numpy as np
    
    if size is None:
        size = 50
        
    # Simple sample data generation
    texts = []
    labels = []
    
    for i in range(size):
        if i % 4 == 0:
            texts.append(f"Great product {i}! Highly recommend.")
            labels.append(4.5)
        elif i % 4 == 1:
            texts.append(f"Terrible quality {i}, very disappointed.")
            labels.append(1.5)
        elif i % 4 == 2:
            texts.append(f"Average product {i}, nothing special.")
            labels.append(3.0)
        else:
            texts.append(f"Good value {i}, satisfied with purchase.")
            labels.append(4.0)
            
    return list(zip(texts, labels))

# Initialize test environment when module is imported
setup_test_environment()
