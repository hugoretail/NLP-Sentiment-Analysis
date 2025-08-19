#!/usr/bin/env python3
"""
Setup script for the NLP Sentiment Analysis project.
Automates the installation and configuration process.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import urllib.request
import zipfile
import shutil

def print_step(step, message):
    """Print a formatted step message."""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {message}")
    print('='*60)

def print_info(message):
    """Print an info message."""
    print(f"ℹ️  {message}")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        import pip
        print_success("pip is available")
        return True
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "ensurepip"], check=True)
            print_success("pip installed successfully")
            return True
        except subprocess.CalledProcessError:
            print_error("Failed to install pip")
            return False

def install_requirements():
    """Install Python requirements."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False
    
    try:
        print_info("Installing Python dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "data",
        "logs",
        "temp",
        "results",
        "tests/test_data",
        "tests/test_models",
        "tests/test_results"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {directory}")

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        print_info("Downloading NLTK data...")
        
        # Download required NLTK datasets
        datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for dataset in datasets:
            try:
                nltk.download(dataset, quiet=True)
                print_success(f"Downloaded NLTK dataset: {dataset}")
            except Exception as e:
                print_warning(f"Failed to download {dataset}: {e}")
        
        return True
    except ImportError:
        print_warning("NLTK not installed, skipping NLTK data download")
        return False

def create_sample_config():
    """Create sample configuration files."""
    # Create local config file
    config_content = '''"""
Local configuration file for development.
Copy this file and modify as needed.
"""

# Development settings
DEBUG = True
TESTING = False

# Model settings
DEFAULT_MODEL = "mock"  # Use "mock" for testing without real models
MODEL_PATH = "./models"

# API settings
API_RATE_LIMIT = 1000  # Higher limit for development
REQUIRE_API_KEY = False

# Processing settings
MAX_TEXT_LENGTH = 512
BATCH_SIZE = 32
ENABLE_CACHING = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/app.log"

# Database (optional)
DATABASE_URL = None

# Redis (optional)
REDIS_URL = None
'''
    
    config_file = Path("config_local.py")
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write(config_content)
        print_success("Created config_local.py")
    else:
        print_info("config_local.py already exists")

def setup_git_hooks():
    """Set up Git hooks for development."""
    git_dir = Path(".git")
    if not git_dir.exists():
        print_warning("Not a Git repository, skipping Git hooks setup")
        return
    
    # Create pre-commit hook
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    
    pre_commit_hook = hooks_dir / "pre-commit"
    hook_content = '''#!/bin/sh
# Pre-commit hook for NLP Sentiment Analysis

echo "Running pre-commit checks..."

# Run tests
python -m pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Check code style
python -m flake8 --max-line-length=88 --ignore=E203,W503 .
if [ $? -ne 0 ]; then
    echo "Code style check failed. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
exit 0
'''
    
    with open(pre_commit_hook, 'w') as f:
        f.write(hook_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(pre_commit_hook, 0o755)
    
    print_success("Git pre-commit hook created")

def run_tests():
    """Run the test suite to verify installation."""
    try:
        print_info("Running test suite...")
        result = subprocess.run([
            sys.executable, "tests/test_all.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("All tests passed!")
            return True
        else:
            print_error("Some tests failed")
            print("Test output:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print_error(f"Failed to run tests: {e}")
        return False

def setup_environment():
    """Set up environment variables."""
    env_file = Path(".env")
    if not env_file.exists():
        env_content = '''# Environment variables for NLP Sentiment Analysis

# Flask settings
FLASK_ENV=development
FLASK_DEBUG=1

# Model settings
MODEL_TYPE=mock
MODEL_PATH=./models

# API settings
API_RATE_LIMIT=1000

# Logging
LOG_LEVEL=INFO
'''
        with open(env_file, 'w') as f:
            f.write(env_content)
        print_success("Created .env file")
    else:
        print_info(".env file already exists")

def main():
    """Main setup function."""
    print("🎯 NLP Sentiment Analysis - Setup Script")
    print("This script will set up your development environment")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check pip
    print_step(2, "Checking pip installation")
    if not check_pip():
        sys.exit(1)
    
    # Step 3: Install requirements
    print_step(3, "Installing Python dependencies")
    if not install_requirements():
        print_warning("Failed to install some dependencies. You may need to install them manually.")
    
    # Step 4: Create directories
    print_step(4, "Creating project directories")
    create_directories()
    
    # Step 5: Download NLTK data
    print_step(5, "Setting up NLTK data")
    download_nltk_data()
    
    # Step 6: Create configuration files
    print_step(6, "Creating configuration files")
    create_sample_config()
    setup_environment()
    
    # Step 7: Set up Git hooks
    print_step(7, "Setting up Git hooks")
    setup_git_hooks()
    
    # Step 8: Run tests
    print_step(8, "Running test suite")
    if not run_tests():
        print_warning("Some tests failed. The basic setup is complete, but you may need to troubleshoot.")
    
    # Final message
    print_step("COMPLETE", "Setup finished!")
    print("\n🎉 Your NLP Sentiment Analysis project is ready!")
    print("\nNext steps:")
    print("1. Start the application: python app.py")
    print("2. Run the demo: python demo.py")
    print("3. Open your browser to: http://localhost:5000")
    print("4. Read the documentation: docs/user_guide.md")
    print("\nFor training models:")
    print("- python scripts/train_model.py --help")
    print("\nFor running tests:")
    print("- python tests/test_all.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)
