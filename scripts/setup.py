#!/usr/bin/env python3
"""
Setup script for the NLP Sentiment Analysis project.

This script handles the installation and configuration of the sentiment
analysis system, including dependencies, models, and environment setup.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project information
PROJECT_NAME = "NLP-Sentiment-Analysis"
PROJECT_VERSION = "1.0.0"
PYTHON_MIN_VERSION = (3, 7)

# Required packages
REQUIRED_PACKAGES = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "transformers>=4.18.0",
    "torch>=1.11.0",
    "tokenizers>=0.12.0",
    "tqdm>=4.62.0",
    "nltk>=3.7",
    "emoji>=1.7.0",
    "requests>=2.27.0",
    "joblib>=1.1.0",
    "psutil>=5.9.0"
]

# Optional packages for enhanced functionality
OPTIONAL_PACKAGES = [
    "tensorboard>=2.8.0",  # For training visualization
    "wandb>=0.12.0",       # For experiment tracking
    "plotly>=5.6.0",       # For interactive plots
    "streamlit>=1.8.0",    # For web interface
    "fastapi>=0.75.0",     # For API deployment
    "uvicorn>=0.17.0",     # For API server
    "pytest>=7.0.0",       # For testing
    "black>=22.0.0",       # For code formatting
    "flake8>=4.0.0",       # For linting
]

# Pre-trained models to download
DEFAULT_MODELS = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "distilbert-base-uncased-finetuned-sst-2-english"
]


class SetupManager:
    """Manages the setup process for the sentiment analysis system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.cache_dir = self.project_root / ".cache"
        
    def check_python_version(self) -> bool:
        """Check if Python version meets minimum requirements."""
        current_version = sys.version_info[:2]
        if current_version < PYTHON_MIN_VERSION:
            logger.error(
                f"Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}+ required, "
                f"but found {current_version[0]}.{current_version[1]}"
            )
            return False
        
        logger.info(f"Python version {current_version[0]}.{current_version[1]} ✅")
        return True
    
    def create_directories(self) -> None:
        """Create necessary project directories."""
        logger.info("Creating project directories...")
        
        directories = [
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "external",
            self.models_dir,
            self.models_dir / "pretrained",
            self.models_dir / "custom",
            self.models_dir / "checkpoints",
            self.cache_dir,
            self.project_root / "logs",
            self.project_root / "outputs",
            self.project_root / "notebooks" / "experiments"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {directory}")
    
    def install_packages(self, packages: List[str], optional: bool = False) -> bool:
        """Install Python packages using pip."""
        package_type = "optional" if optional else "required"
        logger.info(f"Installing {package_type} packages...")
        
        for package in packages:
            try:
                logger.info(f"  Installing {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"  ✅ {package}")
            except subprocess.CalledProcessError as e:
                if optional:
                    logger.warning(f"  ⚠️ Failed to install optional package {package}: {e}")
                else:
                    logger.error(f"  ❌ Failed to install required package {package}: {e}")
                    return False
        
        return True
    
    def setup_nltk_data(self) -> None:
        """Download required NLTK data."""
        logger.info("Setting up NLTK data...")
        
        try:
            import nltk
            
            # Download required NLTK data
            nltk_downloads = [
                'punkt',
                'stopwords',
                'vader_lexicon',
                'wordnet',
                'omw-1.4'
            ]
            
            for item in nltk_downloads:
                try:
                    nltk.download(item, quiet=True)
                    logger.info(f"  ✅ Downloaded {item}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to download {item}: {e}")
                    
        except ImportError:
            logger.warning("NLTK not installed, skipping NLTK data setup")
    
    def download_models(self, models: List[str]) -> None:
        """Download pre-trained models."""
        logger.info("Downloading pre-trained models...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            for model_name in models:
                try:
                    logger.info(f"  Downloading {model_name}...")
                    
                    # Download tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Download model
                    model = AutoModel.from_pretrained(model_name)
                    
                    # Save to local cache
                    model_path = self.models_dir / "pretrained" / model_name.replace("/", "_")
                    model_path.mkdir(parents=True, exist_ok=True)
                    
                    tokenizer.save_pretrained(model_path)
                    model.save_pretrained(model_path)
                    
                    logger.info(f"  ✅ {model_name}")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to download {model_name}: {e}")
                    
        except ImportError:
            logger.warning("Transformers not installed, skipping model downloads")
    
    def create_config_files(self) -> None:
        """Create default configuration files."""
        logger.info("Creating configuration files...")
        
        # Main config file
        config_content = """# NLP Sentiment Analysis Configuration
# This file contains the main configuration settings for the system

[model]
default_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
cache_dir = "models/pretrained"
max_length = 512
batch_size = 32

[preprocessing]
lowercase = true
remove_urls = true
remove_html = true
remove_emails = true
handle_emojis = true
expand_contractions = true

[inference]
use_cache = true
cache_size = 1000
confidence_threshold = 0.5

[logging]
level = "INFO"
file = "logs/sentiment_analysis.log"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

[performance]
use_gpu = true
num_workers = 4
pin_memory = true
"""
        
        config_path = self.project_root / "config.ini"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"  ✅ Created {config_path}")
        
        # Environment template file
        env_content = """# Environment variables for NLP Sentiment Analysis
# Copy this file to .env and update with your settings

# Hugging Face token (optional, for private models)
HUGGINGFACE_TOKEN=your_token_here

# Weights & Biases API key (optional, for experiment tracking)
WANDB_API_KEY=your_api_key_here

# CUDA settings
CUDA_VISIBLE_DEVICES=0

# Cache directories
TRANSFORMERS_CACHE=./models/pretrained
HF_DATASETS_CACHE=./data/cache

# Logging
LOG_LEVEL=INFO
"""
        
        env_path = self.project_root / ".env.template"
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info(f"  ✅ Created {env_path}")
    
    def verify_installation(self) -> bool:
        """Verify that the installation was successful."""
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            import numpy as np
            import pandas as pd
            import sklearn
            import transformers
            import torch
            
            logger.info("  ✅ Core packages imported successfully")
            
            # Test GPU availability
            if torch.cuda.is_available():
                logger.info(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("  ℹ️ CUDA not available, using CPU")
            
            # Test our modules
            sys.path.insert(0, str(self.project_root))
            
            from utils.text_preprocessor import TextPreprocessor
            from utils.model_utils import SentimentModel
            
            logger.info("  ✅ Custom modules imported successfully")
            
            # Quick functionality test
            preprocessor = TextPreprocessor()
            test_text = "This is a test! 😊"
            cleaned = preprocessor.clean_text(test_text)
            
            logger.info("  ✅ Text preprocessing test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Verification failed: {e}")
            return False
    
    def run_setup(self, args: argparse.Namespace) -> bool:
        """Run the complete setup process."""
        logger.info(f"Starting setup for {PROJECT_NAME} v{PROJECT_VERSION}")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create directories
        self.create_directories()
        
        # Install required packages
        if not self.install_packages(REQUIRED_PACKAGES):
            logger.error("Failed to install required packages")
            return False
        
        # Install optional packages if requested
        if args.install_optional:
            self.install_packages(OPTIONAL_PACKAGES, optional=True)
        
        # Setup NLTK data
        self.setup_nltk_data()
        
        # Download models if requested
        if args.download_models:
            models_to_download = args.models if args.models else DEFAULT_MODELS
            self.download_models(models_to_download)
        
        # Create configuration files
        self.create_config_files()
        
        # Verify installation
        if args.verify and not self.verify_installation():
            logger.error("Installation verification failed")
            return False
        
        logger.info("✅ Setup completed successfully!")
        logger.info(f"Project is ready at: {self.project_root}")
        
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup script for NLP Sentiment Analysis project"
    )
    
    parser.add_argument(
        '--install-optional',
        action='store_true',
        help='Install optional packages for enhanced functionality'
    )
    
    parser.add_argument(
        '--download-models',
        action='store_true',
        help='Download pre-trained models'
    )
    
    parser.add_argument(
        '--models',
        nargs='*',
        help='Specific models to download (default: use DEFAULT_MODELS)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Verify installation after setup (default: True)'
    )
    
    parser.add_argument(
        '--no-verify',
        dest='verify',
        action='store_false',
        help='Skip installation verification'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run setup
    setup_manager = SetupManager()
    success = setup_manager.run_setup(args)
    
    if success:
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your NLP Sentiment Analysis system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python scripts/demo.py' to see the system in action")
        print("2. Run 'python scripts/test_model.py' to test the models")
        print("3. Check the documentation in the docs/ directory")
        print("4. Start building your own sentiment analysis applications!")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ SETUP FAILED")
        print("="*60)
        print("Please check the error messages above and try again.")
        print("You may need to install dependencies manually or check your Python environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
