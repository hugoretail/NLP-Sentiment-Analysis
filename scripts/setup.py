# -*- coding: utf-8 -*-
"""Setup script for the sentiment analysis project."""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "models/trained",
        "data/raw", 
        "data/processed",
        "logs",
        "outputs"
    ]
ECHO is on.
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up NLP Sentiment Analysis project...")
ECHO is on.
    create_directories()
ECHO is on.
    if install_requirements():
        print("�� Setup completed successfully!")
        print("Run 'python app.py' to start the application.")
    else:
        print("❌ Setup failed. Please install requirements manually.")

if __name__ == "__main__":
    main()
