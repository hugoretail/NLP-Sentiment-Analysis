# -*- coding: utf-8 -*-
"""Simple test script for the sentiment analysis app."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.inference import SentimentInference

def test_basic_functionality():
    """Test basic sentiment analysis functionality."""
    print("🧪 Testing Sentiment Analysis...")
ECHO is on.
    inference = SentimentInference()
ECHO is on.
    # Test cases
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "Absolutely fantastic experience!",
        "Worst purchase ever, completely disappointed."
    ]
ECHO is on.
    for text in test_texts:
        result = inference.predict_single(text)
        print(f"Text: {text}")
        print(f"Score: {result['score']} stars ({result['sentiment']}^)")
        print(f"Confidence: {result['confidence']}")
        print("-" * 50)
ECHO is on.
    print("✅ Tests completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()
