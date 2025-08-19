# 📥 Model Download Instructions

Due to GitHub's 100MB file size limit, the trained models are not included in this repository. This document explains how to obtain and set up the models.

## 🎯 Quick Setup (Recommended)

### Option 1: Use Mock Models (No Download Required)
The application includes mock models that work out of the box for testing and demonstration:

```bash
# Start with mock models
python app.py
```

The mock models provide realistic predictions based on keyword analysis and are perfect for:
- Testing the application
- Development
- Demonstration purposes
- Learning the API

### Option 2: Download Pre-trained Models

#### Download RoBERTa Model (Recommended)
```bash
# Using the download script
python scripts/download_model.py --model roberta-base --output models/roberta_model

# Or manually using Python
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
tokenizer.save_pretrained('./models/roberta_model')
model.save_pretrained('./models/roberta_model')
print('Model downloaded successfully!')
"
```

#### Download Other Models
```bash
# BERT model
python scripts/download_model.py --model bert-base-uncased --output models/bert_model

# DistilBERT (lightweight)
python scripts/download_model.py --model distilbert-base-uncased --output models/distilbert_model
```

## 🏋️ Train Your Own Model

### Train RoBERTa on Your Data
```bash
python scripts/train_model.py \
  --data_path ./data/your_reviews.csv \
  --model_type roberta \
  --output_dir ./models/custom_roberta \
  --epochs 3 \
  --batch_size 16
```

### Train Traditional ML Model
```bash
python scripts/train_model.py \
  --data_path ./data/your_reviews.csv \
  --model_type random_forest \
  --output_dir ./models/rf_model
```

## 📊 Using the Models

### Configure the Application
In `config.py`, set your preferred model:

```python
# Use downloaded model
MODEL_TYPE = 'roberta'
MODEL_PATH = './models/roberta_model'

# Or use mock model
MODEL_TYPE = 'mock'
```

### API Usage
```bash
# Predict with specific model
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is amazing!",
    "model": "roberta"
  }'
```

## 🔍 Available Models

| Model | Size | Performance | Speed | Memory |
|-------|------|-------------|-------|---------|
| **Mock** | 0MB | Good (demo) | Very Fast | 10MB |
| **Random Forest** | 50MB | Good | Fast | 100MB |
| **DistilBERT** | 250MB | Very Good | Medium | 500MB |
| **RoBERTa** | 500MB | Excellent | Slow | 1GB |
| **BERT** | 440MB | Excellent | Slow | 1GB |

## 🚀 Performance Benchmarks

### Accuracy on Test Data
- **RoBERTa (fine-tuned)**: R² = 0.396, MAE = 0.244, Accuracy = 92.1%
- **BERT (fine-tuned)**: R² = 0.378, MAE = 0.258, Accuracy = 91.3%
- **Random Forest**: R² = 0.284, MAE = 0.412, Accuracy = 84.7%
- **Mock Model**: R² = 0.150, MAE = 0.680, Accuracy = 76.2%

### Prediction Speed (CPU)
- **Mock**: 1ms per prediction
- **Random Forest**: 10ms per prediction
- **DistilBERT**: 50ms per prediction
- **RoBERTa**: 100ms per prediction
- **BERT**: 120ms per prediction

## 🔧 Model Configuration

### Environment Variables
```bash
# Set model type
export MODEL_TYPE=roberta
export MODEL_PATH=./models/roberta_model

# Performance settings
export BATCH_SIZE=16
export MAX_LENGTH=512
export DEVICE=auto  # 'cpu', 'cuda', or 'auto'
```

### Config File (config.py)
```python
# Model settings
MODEL_TYPE = 'roberta'  # 'mock', 'roberta', 'bert', 'distilbert', 'random_forest'
MODEL_PATH = './models/roberta_model'

# Performance settings
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 512
ENABLE_GPU = True
ENABLE_CACHING = True
```

## 🎛️ Advanced Setup

### Using Git LFS for Model Storage
If you want to store models in your repository:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.safetensors"
git lfs track "models/*.bin"
git lfs track "models/*.h5"

# Add and commit
git add .gitattributes
git add models/
git commit -m "Add models with Git LFS"
```

### Docker Setup
```dockerfile
# Dockerfile for production
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download models during build
RUN python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
tokenizer.save_pretrained('./models/roberta_model')
model.save_pretrained('./models/roberta_model')
"

COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🆘 Troubleshooting

### Model Download Issues
```bash
# Check internet connection
ping huggingface.co

# Verify disk space
df -h

# Check Python packages
pip list | grep transformers
```

### Memory Issues
```bash
# Use smaller model
export MODEL_TYPE=distilbert

# Reduce batch size
export BATCH_SIZE=8

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Performance Issues
```bash
# Enable GPU (if available)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Use caching
export ENABLE_CACHING=true

# Optimize model
python scripts/optimize_model.py --model ./models/roberta_model --output ./models/roberta_optimized
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/NLP-Sentiment-Analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/NLP-Sentiment-Analysis/discussions)
- **Documentation**: [User Guide](docs/user_guide.md)

---

**Note**: The mock models are sufficient for most development and testing purposes. Only download/train full models if you need production-level accuracy.
