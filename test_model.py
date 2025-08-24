"""
Test script to verify model loading and basic functionality.
"""

import sys
import os
sys.path.append('src')

def test_model_loading():
    """Test if the sentiment analysis model can be loaded and used."""
    try:
        from src.model_utils import SentimentPredictor
        
        print("🔄 Loading sentiment analysis model...")
        predictor = SentimentPredictor('models')
        
        print("✅ Model loaded successfully!")
        
        test_text = "I love this product! It's amazing."
        result = predictor.predict_single(test_text)
        
        print(f"\n📝 Test Prediction:")
        print(f"Text: {test_text}")
        print(f"Result: {result}")
        
        model_info = predictor.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"Model name: {model_info['model_name']}")
        print(f"Device: {model_info['device']}")
        print(f"Parameters: {model_info['num_parameters']:,}")
        
        metrics = model_info['performance_metrics']
        print(f"\n🎯 Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        
        print(f"\n✅ All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_engine():
    """Test the inference engine with caching."""
    try:
        from src.inference import SentimentInferenceEngine
        
        print("\n🔄 Testing inference engine...")
        engine = SentimentInferenceEngine('models')
        
        test_text = "This product is absolutely fantastic!"
        result = engine.analyze_text_sentiment(test_text)
        
        print(f"📝 Detailed Analysis Result:")
        print(f"Text: {test_text}")
        print(f"Score: {result['sentiment_score']}/5.0")
        print(f"Class: {result['prediction_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if 'sentiment_analysis' in result:
            analysis = result['sentiment_analysis']
            print(f"Interpretation: {analysis.get('interpretation', 'N/A')}")
        
        print(f"✅ Inference engine test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Inference engine error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Sentiment Analysis Model Test")
    print("=" * 50)
    
    if not os.path.exists('models'):
        print("❌ Model directory not found!")
        print("Please ensure the trained model files are in the 'models' directory.")
        sys.exit(1)
    
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'model_info.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(f'models/{file}'):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("Please ensure all model files are present.")
        sys.exit(1)
    
    success = True
    success &= test_model_loading()
    success &= test_inference_engine()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("The sentiment analysis application is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
