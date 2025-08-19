#!/usr/bin/env python3
"""
Model download script for the NLP Sentiment Analysis project.
Downloads and sets up pre-trained models from Hugging Face.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def download_transformers_model(model_name, output_path, task_type='regression'):
    """Download a transformers model and tokenizer."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        logging.error("transformers library not installed. Run: pip install transformers")
        return False
    
    try:
        logging.info(f"Downloading model: {model_name}")
        
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_path)
        logging.info("✅ Tokenizer downloaded")
        
        # Download model
        logging.info("Downloading model...")
        if task_type == 'regression':
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                problem_type="regression"
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=5
            )
        
        model.save_pretrained(output_path)
        logging.info("✅ Model downloaded")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "task_type": task_type,
            "num_labels": 1 if task_type == 'regression' else 5,
            "framework": "transformers",
            "downloaded_by": "download_model.py"
        }
        
        with open(output_path / "download_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logging.info(f"✅ Model {model_name} downloaded successfully to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to download model {model_name}: {e}")
        return False

def create_mock_model(output_path):
    """Create a mock model for testing."""
    try:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create mock model info
        mock_info = {
            "model_name": "mock",
            "task_type": "regression",
            "num_labels": 1,
            "framework": "mock",
            "description": "Mock model for testing and demonstration",
            "features": [
                "Keyword-based sentiment analysis",
                "Fast predictions",
                "No dependencies",
                "Good for testing"
            ]
        }
        
        with open(output_path / "model_info.json", "w") as f:
            json.dump(mock_info, f, indent=2)
        
        # Create mock tokenizer config
        mock_tokenizer = {
            "tokenizer_class": "MockTokenizer",
            "vocab_size": 1000,
            "max_length": 512
        }
        
        with open(output_path / "tokenizer_config.json", "w") as f:
            json.dump(mock_tokenizer, f, indent=2)
        
        logging.info(f"✅ Mock model created at {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to create mock model: {e}")
        return False

def list_available_models():
    """List available pre-trained models."""
    models = {
        "transformer_models": {
            "roberta-base": {
                "description": "RoBERTa base model (125M params)",
                "size": "~500MB",
                "performance": "Excellent",
                "speed": "Slow",
                "recommended": True
            },
            "bert-base-uncased": {
                "description": "BERT base uncased (110M params)",
                "size": "~440MB", 
                "performance": "Excellent",
                "speed": "Slow",
                "recommended": True
            },
            "distilbert-base-uncased": {
                "description": "DistilBERT base uncased (66M params)",
                "size": "~250MB",
                "performance": "Very Good",
                "speed": "Medium",
                "recommended": True
            },
            "albert-base-v2": {
                "description": "ALBERT base v2 (12M params)",
                "size": "~45MB",
                "performance": "Good",
                "speed": "Fast",
                "recommended": False
            }
        },
        "mock_models": {
            "mock": {
                "description": "Mock model for testing",
                "size": "~1KB",
                "performance": "Demo quality",
                "speed": "Very Fast",
                "recommended": True
            }
        }
    }
    
    print("\n📋 Available Models:")
    print("=" * 50)
    
    for category, model_list in models.items():
        print(f"\n📁 {category.replace('_', ' ').title()}:")
        for name, info in model_list.items():
            recommended = " ⭐ RECOMMENDED" if info["recommended"] else ""
            print(f"  🤖 {name}{recommended}")
            print(f"     {info['description']}")
            print(f"     Size: {info['size']}, Performance: {info['performance']}, Speed: {info['speed']}")
    
    print("\n💡 Recommendations:")
    print("  • For production: roberta-base or bert-base-uncased")
    print("  • For development: distilbert-base-uncased")
    print("  • For testing: mock")
    print("  • For low memory: albert-base-v2 or mock")

def check_disk_space(required_gb=2):
    """Check if there's enough disk space."""
    try:
        import shutil
        free_space = shutil.disk_usage(".").free / (1024**3)  # GB
        
        if free_space < required_gb:
            logging.warning(f"⚠️ Low disk space: {free_space:.1f}GB available, {required_gb}GB recommended")
            return False
        else:
            logging.info(f"✅ Sufficient disk space: {free_space:.1f}GB available")
            return True
    except Exception:
        logging.warning("Could not check disk space")
        return True

def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download models for NLP Sentiment Analysis")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name to download (e.g., roberta-base, bert-base-uncased, mock)")
    parser.add_argument("--output", type=str, default="./models",
                       help="Output directory for the model")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"],
                       help="Task type for the model")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force download even if model exists")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.list:
        list_available_models()
        return True
    
    # Check disk space
    if not check_disk_space():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logging.info("Download cancelled by user")
            return False
    
    # Determine output path
    if args.model == "mock":
        output_path = Path(args.output) / "mock_model"
    else:
        model_name_safe = args.model.replace("/", "_").replace("-", "_")
        output_path = Path(args.output) / f"{model_name_safe}_model"
    
    # Check if model already exists
    if output_path.exists() and not args.force:
        logging.warning(f"Model already exists at {output_path}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            logging.info("Download cancelled by user")
            return False
    
    # Download model
    if args.model == "mock":
        success = create_mock_model(output_path)
    else:
        success = download_transformers_model(args.model, output_path, args.task)
    
    if success:
        print(f"\n🎉 Success! Model downloaded to: {output_path}")
        print("\n📝 Next steps:")
        print(f"  1. Update config.py: MODEL_PATH = '{output_path}'")
        print(f"  2. Update config.py: MODEL_TYPE = '{args.model.split('-')[0] if args.model != 'mock' else 'mock'}'")
        print("  3. Start the application: python app.py")
        print("  4. Test the model: python -c \"from app import app; print('Model loaded successfully!')\"")
    else:
        print("\n❌ Download failed. Check the logs above for details.")
        return False
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
