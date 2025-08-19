"""
Test configuration file.
Contains settings and fixtures for the test suite.
"""

import os
import sys
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    'data': {
        'sample_size': 100,
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42
    },
    'models': {
        'sklearn': {
            'test_models': ['random_forest', 'ridge', 'svr'],
            'default_params': {
                'random_forest': {'n_estimators': 10, 'random_state': 42},
                'ridge': {'alpha': 1.0},
                'svr': {'kernel': 'rbf', 'C': 1.0}
            }
        },
        'transformers': {
            'test_models': ['distilbert-base-uncased'],  # Smaller model for testing
            'max_length': 128,
            'batch_size': 8,
            'epochs': 1
        }
    },
    'evaluation': {
        'metrics': ['mae', 'rmse', 'r2', 'accuracy', 'f1_macro'],
        'tolerance': {
            'mae': 0.1,
            'rmse': 0.1,
            'r2': 0.05,
            'accuracy': 0.05
        }
    },
    'paths': {
        'test_data_dir': 'test_data',
        'test_models_dir': 'test_models',
        'test_results_dir': 'test_results'
    }
}

# Sample test data
SAMPLE_REVIEWS = [
    ("This product is absolutely amazing! I love everything about it.", 5),
    ("Terrible quality, complete waste of money. Very disappointed.", 1),
    ("Good value for money, would recommend to others.", 4),
    ("Average product, nothing special but does the job.", 3),
    ("Excellent service and very fast delivery! Highly recommended!", 5),
    ("Poor quality control, arrived damaged.", 2),
    ("Great features and easy to use interface.", 4),
    ("Overpriced for what you get, not worth it.", 2),
    ("Perfect for my needs, exactly what I wanted.", 5),
    ("Customer service was unhelpful and rude.", 1),
    ("Decent product, meets expectations.", 3),
    ("Outstanding quality and fast shipping!", 5),
    ("Not bad, but could be better for the price.", 3),
    ("Horrible experience, would not buy again.", 1),
    ("Really good value, happy with purchase.", 4),
    ("Mediocre quality, expected more.", 2),
    ("Fantastic product, exceeded expectations!", 5),
    ("Okay product, nothing to write home about.", 3),
    ("Very poor quality, disappointed.", 1),
    ("Great purchase, highly recommend!", 4)
]

# Test utility functions
def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

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
    
    size = size or TEST_CONFIG['data']['sample_size']
    np.random.seed(TEST_CONFIG['data']['random_state'])
    
    # Repeat sample reviews to create larger dataset
    reviews = []
    ratings = []
    
    for i in range(size):
        review, rating = SAMPLE_REVIEWS[i % len(SAMPLE_REVIEWS)]
        # Add some variation
        if np.random.random() < 0.1:  # 10% chance to add noise
            rating = max(1, min(5, rating + np.random.choice([-1, 1])))
        
        reviews.append(review)
        ratings.append(rating)
    
    return pd.DataFrame({
        'review': reviews,
        'rating': ratings
    })

def setup_test_environment():
    """Set up the test environment."""
    # Create test directories
    test_dirs = [
        get_test_data_path(),
        get_test_models_path(),
        get_test_results_path()
    ]
    
    for test_dir in test_dirs:
        test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample test data
    sample_df = create_sample_dataset()
    sample_df.to_csv(get_test_data_path() / 'sample_data.csv', index=False)
    
    print("Test environment set up successfully!")

def cleanup_test_environment():
    """Clean up the test environment."""
    import shutil
    
    test_dirs = [
        get_test_data_path(),
        get_test_models_path(),
        get_test_results_path()
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
    
    print("Test environment cleaned up!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_test_environment()
    elif len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_test_environment()
    else:
        print("Usage: python test_config.py [setup|cleanup]")
