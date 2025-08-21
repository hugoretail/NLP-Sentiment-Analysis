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

## 🎮 User Guide

### Single Text Analysis
1. Navigate to the "Analyze" tab
2. Enter or paste your text
3. Click "Analyze Sentiment"
4. View results with star rating, confidence, and interpretation

### Batch Processing
1. Go to the "Batch" tab
2. Enter multiple texts (one per line)
3. Click "Analyze Batch"
4. Review summary statistics and detailed results

### Text Comparison
1. Switch to the "Compare" tab
2. Add multiple texts with custom labels
3. Click "Compare"
4. Analyze side-by-side results and charts

### Dashboard Analytics
1. Visit the "Dashboard" tab
2. View real-time analytics and insights
3. Explore sentiment distributions and trends
4. Review analysis history

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

### Running in Development Mode

```bash
# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=1

# Run with auto-reload
python app/app.py
```

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

## 🔧 Configuration

### Model Configuration
Edit `models/config.json` to modify model parameters:

```json
{
  "model_type": "roberta",
  "num_labels": 1,
  "max_position_embeddings": 514,
  "hidden_size": 768
}
```

### Application Configuration
Modify settings in `app/app.py`:

```python
# Cache settings
CACHE_SIZE = 1000
ENABLE_CACHE = True

# Model settings
MODEL_PATH = "../models"
BATCH_SIZE = 16
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**:
```bash
pip install gunicorn
export FLASK_ENV=production
```

2. **Run with Gunicorn**:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Docker Deployment**:
```dockerfile
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Performance Optimization

- **GPU Support**: Ensure CUDA is available for faster inference
- **Model Caching**: Results are cached to improve response times
- **Batch Processing**: Use batch endpoints for multiple texts
- **Load Balancing**: Use multiple workers for high traffic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📚 Technical Details

### Model Architecture
- **Base**: RoBERTa-base transformer
- **Task**: Regression (continuous scores 1-5)
- **Output**: Single scalar value with sigmoid activation
- **Loss**: Mean Squared Error with normalization

### Text Preprocessing
- URL and mention removal
- Special character handling
- Length normalization
- Optional stopword removal

### Inference Optimization
- **Caching**: LRU cache for repeated texts
- **Batching**: Efficient batch processing
- **Threading**: Thread-safe operations
- **Memory Management**: Automatic cleanup

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure model files are in the correct directory
   - Check file permissions and disk space

2. **CUDA Out of Memory**:
   - Reduce batch size in configuration
   - Use CPU inference if GPU memory is insufficient

3. **Slow Inference**:
   - Enable caching in configuration
   - Use batch processing for multiple texts
   - Consider GPU acceleration

4. **Import Errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

### Performance Monitoring

Monitor the application with built-in metrics:
- Response times
- Cache hit rates
- Memory usage
- Error rates

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model infrastructure
- [RoBERTa](https://arxiv.org/abs/1907.11692) for the base model architecture
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the UI components
- [Chart.js](https://www.chartjs.org/) for visualizations

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Made with ❤️ for accurate sentiment analysis**
