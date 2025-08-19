#!/usr/bin/env python3
"""
Demonstration script for the NLP Sentiment Analysis project.
Shows all major features and capabilities.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.data_processor import DataProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.evaluation import calculate_metrics

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📋 {title}")
    print("-" * 40)

def create_demo_data():
    """Create demonstration data."""
    demo_reviews = [
        ("This product is absolutely amazing! Best purchase ever!", 5),
        ("Terrible quality, complete waste of money.", 1),
        ("Good value for money, would recommend.", 4),
        ("Average product, nothing special.", 3),
        ("Excellent service and fast delivery!", 5),
        ("Poor quality control, arrived damaged.", 2),
        ("Great features, easy to use.", 4),
        ("Overpriced for what you get.", 2),
        ("Perfect for my needs!", 5),
        ("Customer service was unhelpful.", 1),
        ("Decent product, meets expectations.", 3),
        ("Outstanding quality!", 5),
        ("Not bad, but could be better.", 3),
        ("Horrible experience, wouldn't buy again.", 1),
        ("Really good value for money.", 4),
        ("Mediocre quality, expected more.", 2),
        ("Fantastic product, exceeded expectations!", 5),
        ("Okay product, nothing special.", 3),
        ("Very poor quality.", 1),
        ("Highly recommend this product!", 4)
    ]
    
    # Create more data by adding variations
    extended_reviews = []
    for review, rating in demo_reviews:
        extended_reviews.append((review, rating))
        # Add some variations
        extended_reviews.append((review + " Great experience overall.", rating))
        extended_reviews.append((review + " Would consider buying again.", rating))
    
    return pd.DataFrame(extended_reviews, columns=['review', 'rating'])

def demo_data_processing():
    """Demonstrate data processing capabilities."""
    print_header("📊 DATA PROCESSING DEMONSTRATION")
    
    # Create demo data
    print_section("Creating Demo Dataset")
    df = create_demo_data()
    print(f"✅ Created dataset with {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Sample: {df.iloc[0]['review'][:50]}...")
    
    # Initialize data processor
    print_section("Data Processor Initialization")
    data_processor = DataProcessor()
    data_processor.df = df.copy()
    print("✅ Data processor initialized")
    
    # Preprocess data
    print_section("Data Preprocessing")
    processed_df = data_processor.preprocess_data()
    print(f"✅ Preprocessed {len(processed_df)} samples")
    
    # Get statistics
    print_section("Data Statistics")
    stats = data_processor.get_data_statistics()
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Rating distribution: {stats['rating_distribution']}")
    print(f"   Average text length: {stats['text_length_stats']['mean']:.1f} words")
    
    # Split data
    print_section("Data Splitting")
    train_data, val_data, test_data = data_processor.split_data(
        test_size=0.2, val_size=0.1
    )
    print(f"   Train samples: {len(train_data['texts'])}")
    print(f"   Validation samples: {len(val_data['texts'])}")
    print(f"   Test samples: {len(test_data['texts'])}")
    
    return train_data, val_data, test_data

def demo_text_preprocessing():
    """Demonstrate text preprocessing capabilities."""
    print_header("🔤 TEXT PREPROCESSING DEMONSTRATION")
    
    # Initialize preprocessor
    print_section("Text Preprocessor Initialization")
    preprocessor = TextPreprocessor()
    print("✅ Text preprocessor initialized")
    
    # Test text cleaning
    print_section("Text Cleaning Examples")
    test_texts = [
        "I can't believe this product is AMAZING!!! 😍",
        "Visit https://example.com for more info",
        "<p>This is <b>bold</b> HTML text.</p>",
        "Email me@test.com about this issue!!!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        cleaned = preprocessor.preprocess_text(text)
        print(f"   Example {i}:")
        print(f"     Original: {text}")
        print(f"     Cleaned:  {cleaned}")
    
    # Batch processing
    print_section("Batch Text Processing")
    batch_texts = [
        "This product is great!",
        "I don't like it at all.",
        "It's okay, I guess."
    ]
    
    start_time = time.time()
    processed_batch = preprocessor.preprocess_batch(batch_texts)
    processing_time = time.time() - start_time
    
    print(f"✅ Processed {len(processed_batch)} texts in {processing_time:.3f}s")
    for original, processed in zip(batch_texts, processed_batch):
        print(f"   '{original}' → '{processed}'")
    
    # Word frequencies
    print_section("Word Frequency Analysis")
    word_freq = preprocessor.get_word_frequencies(processed_batch, top_n=10)
    print("   Top words:")
    for word, count in word_freq.items():
        print(f"     {word}: {count}")
    
    return preprocessor

def demo_evaluation():
    """Demonstrate evaluation capabilities."""
    print_header("📈 EVALUATION DEMONSTRATION")
    
    # Create mock predictions
    print_section("Creating Mock Predictions")
    np.random.seed(42)
    n_samples = 50
    y_true = np.random.uniform(1, 5, n_samples)
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)
    y_pred = np.clip(y_pred, 1, 5)
    
    print(f"✅ Created {n_samples} mock predictions")
    
    # Calculate metrics
    print_section("Calculating Metrics")
    metrics = calculate_metrics(y_true, y_pred, task_type='regression')
    
    print("   Regression Metrics:")
    print(f"     R² Score: {metrics['r2']:.4f}")
    print(f"     MAE: {metrics['mae']:.4f}")
    print(f"     RMSE: {metrics['rmse']:.4f}")
    
    print("   Classification Metrics:")
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    # Per-class metrics
    print_section("Per-Class Performance")
    for star in [1, 2, 3, 4, 5]:
        precision_key = f'precision_{star}star'
        recall_key = f'recall_{star}star'
        f1_key = f'f1_{star}star'
        
        if all(key in metrics for key in [precision_key, recall_key, f1_key]):
            print(f"   {star} Star - P: {metrics[precision_key]:.3f}, "
                  f"R: {metrics[recall_key]:.3f}, F1: {metrics[f1_key]:.3f}")
    
    return metrics

def demo_integration():
    """Demonstrate full integration."""
    print_header("🔗 INTEGRATION DEMONSTRATION")
    
    # Complete pipeline
    print_section("Complete Pipeline Test")
    
    # 1. Data processing
    print("1️⃣ Processing data...")
    train_data, val_data, test_data = demo_data_processing()
    
    # 2. Text preprocessing
    print("\n2️⃣ Preprocessing text...")
    preprocessor = demo_text_preprocessing()
    
    # 3. Mock model training/prediction
    print("\n3️⃣ Simulating model predictions...")
    test_texts = test_data['texts'][:10]  # Take first 10 for demo
    test_ratings = test_data['ratings'][:10]
    
    # Apply text preprocessing
    processed_texts = preprocessor.preprocess_batch(test_texts)
    
    # Create mock predictions (simulate a trained model)
    np.random.seed(42)
    mock_predictions = []
    for rating in test_ratings:
        # Add some realistic noise
        pred = rating + np.random.normal(0, 0.3)
        pred = np.clip(pred, 1, 5)
        mock_predictions.append(pred)
    
    # 4. Evaluation
    print("\n4️⃣ Evaluating predictions...")
    metrics = calculate_metrics(
        np.array(test_ratings), 
        np.array(mock_predictions), 
        task_type='regression'
    )
    
    print("   Final Results:")
    print(f"     Samples processed: {len(test_texts)}")
    print(f"     R² Score: {metrics['r2']:.4f}")
    print(f"     MAE: {metrics['mae']:.4f}")
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    
    # Show some predictions
    print_section("Sample Predictions")
    for i in range(min(5, len(test_texts))):
        print(f"   Text: {test_texts[i][:50]}...")
        print(f"   True: {test_ratings[i]:.1f}, Predicted: {mock_predictions[i]:.2f}")
        print()

def demo_api_simulation():
    """Demonstrate API functionality simulation."""
    print_header("🌐 API SIMULATION DEMONSTRATION")
    
    print_section("Simulating API Calls")
    
    # Mock API responses
    api_responses = []
    
    test_requests = [
        {"text": "This product is amazing!", "model": "roberta"},
        {"text": "Poor quality, not recommended.", "model": "roberta"},
        {"texts": ["Great!", "Bad!", "Okay."], "batch": True}
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"   Request {i}: {request}")
        
        if "texts" in request:  # Batch request
            response = {
                "predictions": [4.5, 1.8, 3.2],
                "confidences": [0.89, 0.76, 0.65],
                "processing_time": 0.123,
                "batch_size": len(request["texts"])
            }
        else:  # Single request
            # Mock prediction based on simple keyword analysis
            text = request["text"].lower()
            if any(word in text for word in ["amazing", "great", "excellent"]):
                pred = 4.5
                conf = 0.92
            elif any(word in text for word in ["poor", "bad", "terrible"]):
                pred = 1.8
                conf = 0.87
            else:
                pred = 3.0
                conf = 0.65
                
            response = {
                "prediction": pred,
                "confidence": conf,
                "rating_class": round(pred),
                "processing_time": 0.045,
                "model_used": request.get("model", "default")
            }
        
        api_responses.append(response)
        print(f"   Response {i}: {response}")
        print()
    
    print("✅ API simulation completed")

def run_all_demos():
    """Run all demonstration modules."""
    print_header("🎯 NLP SENTIMENT ANALYSIS - COMPLETE DEMONSTRATION")
    print("This demonstration shows all the features and capabilities")
    print("of the NLP Sentiment Analysis project.")
    
    try:
        # Run individual demos
        demo_data_processing()
        demo_text_preprocessing()
        demo_evaluation()
        demo_integration()
        demo_api_simulation()
        
        # Final summary
        print_header("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("All modules are working correctly!")
        print("\nNext steps:")
        print("1. Start the web application: python app.py")
        print("2. Train a model: python scripts/train_model.py --help")
        print("3. Run tests: python tests/test_all.py")
        print("4. Read the docs: docs/user_guide.md")
        
    except Exception as e:
        print_header("❌ DEMONSTRATION FAILED")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify Python version: python --version (3.8+ required)")
        print("3. Run tests to identify issues: python tests/test_all.py")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)
