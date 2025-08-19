# User Guide - NLP Sentiment Analysis

## Table of Contents
1. [Getting Started](#getting-started)
2. [Web Interface Guide](#web-interface-guide)
3. [API Usage](#api-usage)
4. [Model Training](#model-training)
5. [Batch Processing](#batch-processing)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for transformer models)
- 2GB free disk space
- Internet connection for downloading pre-trained models

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/NLP-Sentiment-Analysis.git
cd NLP-Sentiment-Analysis
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up the environment**:
```bash
python tests/test_config.py setup
```

### Quick Start

Run the application:
```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`

## Web Interface Guide

### Main Dashboard

The main dashboard provides several key features:

#### Single Text Prediction
1. Enter your review text in the text area
2. Select a model from the dropdown (if multiple models are available)
3. Click "Analyze Sentiment"
4. View the prediction result with confidence score

#### Real-time Analytics
- **Current Session Stats**: Number of predictions made
- **Model Performance**: Live accuracy metrics
- **Recent Predictions**: History of your recent analyses

### Advanced Features

#### Model Comparison
1. Navigate to the "Analytics" tab
2. Select multiple models for comparison
3. View side-by-side performance metrics
4. Analyze prediction differences

#### Error Analysis
1. Go to "Analytics" → "Error Analysis"
2. Upload test data with known labels
3. View detailed error breakdown by rating class
4. Identify model weaknesses and strengths

### Navigation

- **Home**: Main prediction interface
- **Batch**: Bulk processing for multiple texts
- **Analytics**: Detailed performance analysis
- **API Docs**: Interactive API documentation
- **Models**: Model management and information

## API Usage

### Authentication

Currently, the API is open by default. For production use, enable API key authentication in `config.py`:

```python
REQUIRE_API_KEY = True
API_KEYS = ['your-secret-api-key']
```

### Endpoints

#### Single Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "text": "This product is amazing!",
  "model": "roberta"  # optional
}
```

**Response**:
```json
{
  "prediction": 4.8,
  "confidence": 0.92,
  "rating_class": 5,
  "processing_time": 0.045,
  "model_used": "roberta-base"
}
```

#### Batch Predictions
```bash
POST /api/predict/batch
Content-Type: application/json

{
  "texts": [
    "Great product!",
    "Poor quality.",
    "Average item."
  ],
  "return_details": true
}
```

#### Model Information
```bash
GET /api/models
```

#### Health Check
```bash
GET /api/health
```

### Rate Limiting

Default rate limit: 100 requests per minute per IP. Configure in `config.py`:

```python
API_RATE_LIMIT = 100  # requests per minute
```

### Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

Error response format:
```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "details": "Additional error details"
}
```

## Model Training

### Supported Models

#### Transformer Models
- **RoBERTa**: `roberta-base` (recommended)
- **BERT**: `bert-base-uncased`
- **DistilBERT**: `distilbert-base-uncased` (lightweight)

#### Traditional ML Models
- **Random Forest**: Good baseline performance
- **Ridge Regression**: Fast and efficient
- **Support Vector Regression**: Good for smaller datasets

### Training a New Model

#### Basic Training
```bash
python scripts/train_model.py \
  --data_path ./data/reviews.csv \
  --model_type roberta \
  --output_dir ./models/my_model
```

#### Advanced Training Options
```bash
python scripts/train_model.py \
  --data_path ./data/reviews.csv \
  --model_type roberta \
  --output_dir ./models/custom_roberta \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --test_size 0.2 \
  --val_size 0.1
```

### Data Format

Your training data should be a CSV file with the following columns:

```csv
review,rating
"This product is amazing!",5
"Poor quality, not recommended.",1
"Good value for money.",4
```

Required columns:
- **review**: Text review content
- **rating**: Integer rating from 1 to 5

### Model Evaluation

After training, evaluate your model:

```bash
python scripts/test_model.py \
  --model_path ./models/my_model \
  --model_type transformer \
  --test_data ./data/test_reviews.csv \
  --generate_visualizations \
  --analyze_errors
```

## Batch Processing

### Web Interface Batch Processing

1. Navigate to the "Batch" tab
2. Upload a CSV file with a "review" column
3. Select your preferred model
4. Click "Process Batch"
5. Download results when processing is complete

### Command Line Batch Processing

```bash
python scripts/batch_predict.py \
  --input_file ./data/reviews_to_predict.csv \
  --output_file ./results/predictions.csv \
  --model_path ./models/roberta_model
```

### Batch File Format

Input CSV format:
```csv
id,review
1,"This product is great!"
2,"Not satisfied with quality."
3,"Average product."
```

Output CSV format:
```csv
id,review,prediction,confidence,rating_class
1,"This product is great!",4.2,0.89,4
2,"Not satisfied with quality.",1.8,0.76,2
3,"Average product.",3.1,0.65,3
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

**Error**: `FileNotFoundError: Model not found`

**Solution**:
```bash
# Check if model directory exists
ls -la models/

# Download a pre-trained model
python scripts/download_model.py --model roberta-base
```

#### 2. Memory Issues

**Error**: `CUDA out of memory` or `RuntimeError: out of memory`

**Solutions**:
```bash
# Use smaller batch size
export BATCH_SIZE=8

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Or use a smaller model
# Use distilbert instead of roberta
```

#### 3. NLTK Data Missing

**Error**: `LookupError: Resource punkt not found`

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

#### 4. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 5000
lsof -i :5000  # On Unix/Mac
netstat -ano | findstr :5000  # On Windows

# Kill the process or use different port
export FLASK_PORT=5001
python app.py
```

#### 5. Slow Predictions

**Causes and Solutions**:

- **Large model**: Use a smaller model like DistilBERT
- **CPU-only mode**: Enable GPU if available
- **Large batch size**: Reduce batch size
- **No caching**: Enable prediction caching in config

### Performance Optimization

#### GPU Usage
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi  # If NVIDIA GPU
```

#### Memory Optimization
```python
# In config.py
MAX_SEQUENCE_LENGTH = 256  # Reduce from 512
BATCH_SIZE = 16           # Reduce from 32
ENABLE_CACHING = True     # Cache frequent predictions
```

#### Model Optimization
```bash
# Convert model to ONNX for faster inference
python scripts/convert_to_onnx.py \
  --model_path ./models/roberta_model \
  --output_path ./models/roberta_model.onnx
```

### Getting Help

1. **Check the logs**: Look in `app.log` for error details
2. **Run tests**: `python tests/test_all.py` to verify installation
3. **Check system requirements**: Ensure you meet minimum requirements
4. **Update dependencies**: `pip install -r requirements.txt --upgrade`

### Performance Benchmarks

#### Expected Performance (on modern hardware)

| Model Type | CPU Prediction Time | GPU Prediction Time | Memory Usage |
|------------|-------------------|-------------------|--------------|
| Random Forest | 10ms | N/A | 100MB |
| Ridge Regression | 5ms | N/A | 50MB |
| DistilBERT | 50ms | 15ms | 500MB |
| RoBERTa | 100ms | 25ms | 1GB |
| BERT | 120ms | 30ms | 1.2GB |

#### Batch Processing Performance

| Batch Size | RoBERTa (GPU) | RoBERTa (CPU) | Random Forest |
|------------|---------------|---------------|---------------|
| 1 text | 25ms | 100ms | 10ms |
| 10 texts | 80ms | 900ms | 50ms |
| 100 texts | 600ms | 8s | 200ms |
| 1000 texts | 5s | 80s | 1.5s |

### Contact Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/NLP-Sentiment-Analysis/issues)
- **Discussions**: [Community support and questions](https://github.com/yourusername/NLP-Sentiment-Analysis/discussions)
- **Email**: support@yourproject.com
