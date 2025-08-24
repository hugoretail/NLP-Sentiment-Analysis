# 🤖 Advanced Sentiment Analysis Application

A comprehensive sentiment analysis application built with RoBERTa transformer model, featuring continuous sentiment scoring (1-5 stars), advanced visualizations, and a modern web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🌟 Features

### 🎯 Advanced Sentiment Analysis
- **Continuous Scoring**: 1-5 star rating system (not just positive/negative)
- **High Accuracy**: 74%+ accuracy with R² score of 0.39+
- **Confidence Estimation**: Provides confidence levels for each prediction
- **Fast Inference**: Optimized for real-time predictions with caching

### 🎨 Modern Web Interface
- **Responsive Design**: Mobile-friendly Bootstrap 5 interface
- **Interactive Dashboard**: Real-time analytics and visualizations
- **Batch Processing**: Analyze multiple texts simultaneously
- **Text Comparison**: Side-by-side sentiment comparison
- **History Tracking**: Persistent analysis history with insights

### 📊 Comprehensive Analytics
- **Real-time Charts**: Sentiment distribution and activity tracking
- **Performance Metrics**: Model accuracy, confidence, and error analysis
- **Insights Generation**: Automated insights from analysis patterns
- **Export Capabilities**: JSON reports and data export

### 🛠️ Developer-Friendly
- **Modular Architecture**: Clean, well-documented codebase
- **API Endpoints**: RESTful API for integration
- **Jupyter Notebooks**: Comprehensive evaluation and exploration tools
- **Error Handling**: Robust error handling and logging

## 🏗️ Architecture

```
sentiment-analysis-app/
├── 📁 data/                     # Data storage
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed datasets
│   └── splits/                  # Train/validation/test splits
├── 📁 models/                   # Trained model files
│   ├── config.json              # Model configuration
│   ├── model.safetensors        # Model weights
│   ├── tokenizer.json           # Tokenizer files
│   └── model_info.json          # Training metadata
├── 📁 src/                      # Core modules
│   ├── data_preprocessing.py    # Text preprocessing utilities
│   ├── model_utils.py           # Model loading and utilities
│   ├── inference.py             # Inference engine with caching
│   └── evaluation.py            # Evaluation and metrics
├── 📁 app/                      # Flask web application
│   ├── app.py                   # Main Flask application
│   ├── static/                  # CSS, JS, images
│   └── templates/               # HTML templates
├── 📁 notebooks/                # Jupyter notebooks
│   └── model_evaluation.ipynb   # Comprehensive model analysis
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- 8GB+ RAM (recommended for model loading)
- GPU support (optional, for faster inference)

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-analysis-app

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run the Application

```bash
# Start the Flask web application
cd app
python app.py

# Access the application
# Open http://localhost:5000 in your browser
```

### 3. API Usage

```python
import requests

# Single text analysis
response = requests.post('http://localhost:5000/api/predict', 
                        json={'text': 'I love this product!'})
result = response.json()
print(f"Sentiment: {result['sentiment_score']}/5.0")

# Batch analysis
response = requests.post('http://localhost:5000/api/batch_predict',
                        json={'texts': ['Great!', 'Terrible!', 'Okay.']})
results = response.json()
```

## 📊 Model Performance

### Key Metrics
- **Accuracy**: 74.9% (±0.5 stars)
- **MAE**: 0.243 (Mean Absolute Error)
- **RMSE**: 0.286 (Root Mean Square Error)
- **R² Score**: 0.396 (Coefficient of Determination)
- **Inference Time**: ~50ms per text (CPU)

### Training Details
- **Base Model**: RoBERTa-base (124M parameters)
- **Training Data**: 69,260 samples
- **Validation Data**: 14,842 samples
- **Test Data**: 14,842 samples
- **Epochs**: 3
- **Learning Rate**: 5e-5

## 🔧 API Reference

### Endpoints

#### `POST /api/predict`
Analyze sentiment for a single text.

**Request:**
```json
{
  "text": "Your text here",
  "detailed": true
}
```

**Response:**
```json
{
  "sentiment_score": 4.2,
  "confidence": 0.85,
  "prediction_class": "positive",
  "sentiment_analysis": {
    "sentiment_strength": "positive",
    "confidence_level": "high",
    "interpretation": "This text expresses positive sentiment with high confidence (scored 4.2 out of 5.0)."
  },
  "processing_time": 0.045
}
```

#### `POST /api/batch_predict`
Analyze sentiment for multiple texts.

**Request:**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "batch_size": 16
}
```

**Response:**
```json
{
  "results": [...],
  "batch_statistics": {
    "count": 3,
    "mean_score": 3.2,
    "class_distribution": {
      "positive": 1,
      "neutral": 1,
      "negative": 1
    }
  }
}
```

#### `POST /api/compare`
Compare sentiment of multiple texts.

**Request:**
```json
{
  "texts": ["Text 1", "Text 2"],
  "labels": ["Label 1", "Label 2"]
}
```

#### `GET /api/model_info`
Get model information and performance metrics.

#### `GET /api/history`
Retrieve analysis history with pagination.

#### `GET /api/analytics`
Get analytics and insights from analysis history.

## 📈 Development

### Testing the Model

```python
# Load the inference engine
from src.inference import SentimentInferenceEngine

engine = SentimentInferenceEngine("models")

# Test prediction
result = engine.analyze_text_sentiment("I love this product!")
print(result)
```

### Model Evaluation

Use the provided Jupyter notebook for comprehensive model evaluation:

```bash
# Start Jupyter
jupyter notebook notebooks/model_evaluation.ipynb
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
