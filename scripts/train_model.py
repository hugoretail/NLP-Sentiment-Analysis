"""
Training script for sentiment analysis models.
Supports both traditional ML and transformer-based models.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_processor import DataProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.model_utils import ModelUtils, save_sklearn_model
from utils.evaluation import evaluate_model, generate_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_sklearn_model(train_data, val_data, test_data, model_type='random_forest'):
    """Train a scikit-learn model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    
    logger.info(f"Training {model_type} model")
    
    # Create model pipeline
    if model_type == 'random_forest':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    elif model_type == 'ridge':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('regressor', Ridge(alpha=1.0))
        ])
    elif model_type == 'svr':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('regressor', SVR(kernel='rbf'))
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    X_train, y_train = train_data['texts'], train_data['ratings']
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    if val_data['texts']:
        X_val, y_val = val_data['texts'], val_data['ratings']
        val_results = evaluate_model(model, X_val, y_val, task_type='regression', model_type='sklearn')
        logger.info(f"Validation R² Score: {val_results['metrics']['r2']:.4f}")
    
    # Evaluate on test set
    X_test, y_test = test_data['texts'], test_data['ratings']
    test_results = evaluate_model(model, X_test, y_test, task_type='regression', model_type='sklearn')
    
    return model, test_results

def train_transformer_model(train_data, val_data, test_data, model_name='roberta-base'):
    """Train a transformer model."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.metrics import mean_absolute_error, r2_score
    
    logger.info(f"Training {model_name} transformer model")
    
    class SentimentDataset(Dataset):
        def __init__(self, texts, ratings, tokenizer, max_length=512):
            self.texts = texts
            self.ratings = ratings
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            rating = float(self.ratings[idx])
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(rating, dtype=torch.float)
            }
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        return {
            'mae': mae,
            'r2': r2
        }
    
    # Initialize model utils
    model_utils = ModelUtils(model_name)
    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_model(num_labels=1)  # Regression task
    
    # Create datasets
    train_dataset = SentimentDataset(train_data['texts'], train_data['ratings'], tokenizer)
    val_dataset = SentimentDataset(val_data['texts'], val_data['ratings'], tokenizer) if val_data['texts'] else None
    test_dataset = SentimentDataset(test_data['texts'], test_data['ratings'], tokenizer)
    
    # Training arguments
    training_args = model_utils.prepare_training_args(
        output_dir='./models/transformer_training',
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False
    )
    
    # Create trainer
    trainer = model_utils.create_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        early_stopping_patience=2
    )
    
    # Train model
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    # Convert to our evaluation format
    test_predictions = trainer.predict(test_dataset)
    y_pred = test_predictions.predictions.flatten()
    y_true = test_data['ratings']
    
    from utils.evaluation import calculate_metrics
    metrics = calculate_metrics(np.array(y_true), y_pred, task_type='regression')
    
    evaluation_results = {
        'model_type': 'transformers',
        'task_type': 'regression',
        'test_samples': len(y_true),
        'metrics': metrics,
        'predictions': {
            'y_true': y_true,
            'y_pred': y_pred.tolist()
        }
    }
    
    return model, tokenizer, evaluation_results

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_type', type=str, default='roberta', 
                       choices=['roberta', 'bert', 'random_forest', 'ridge', 'svr'],
                       help='Type of model to train')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log training configuration
    logger.info(f"Starting training with configuration: {vars(args)}")
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_processor = DataProcessor()
        df = data_processor.load_data(args.data_path)
        
        # Determine column names
        text_column = 'review' if 'review' in df.columns else df.columns[0]
        rating_column = 'rating' if 'rating' in df.columns else df.columns[1]
        
        logger.info(f"Using columns: text='{text_column}', rating='{rating_column}'")
        
        # Preprocess data
        df_processed = data_processor.preprocess_data(
            text_column=text_column,
            rating_column=rating_column
        )
        
        # Split data
        train_data, val_data, test_data = data_processor.split_data(
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        # Get data statistics
        stats = data_processor.get_data_statistics()
        logger.info(f"Data statistics: {json.dumps(stats, indent=2)}")
        
        # Preprocess text if using sklearn models
        if args.model_type in ['random_forest', 'ridge', 'svr']:
            logger.info("Applying text preprocessing for sklearn models...")
            preprocessor = TextPreprocessor()
            
            train_data['texts'] = preprocessor.preprocess_batch(train_data['texts'])
            if val_data['texts']:
                val_data['texts'] = preprocessor.preprocess_batch(val_data['texts'])
            test_data['texts'] = preprocessor.preprocess_batch(test_data['texts'])
            
            # Save preprocessor
            preprocessor.save_preprocessor(output_dir / 'text_preprocessor.pkl')
        
        # Train model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.model_type}_{timestamp}"
        
        if args.model_type in ['random_forest', 'ridge', 'svr']:
            # Train sklearn model
            model, test_results = train_sklearn_model(train_data, val_data, test_data, args.model_type)
            
            # Save model
            model_path = output_dir / f"{model_name}.joblib"
            model_info = {
                'model_type': args.model_type,
                'training_params': vars(args),
                'data_stats': stats,
                'test_metrics': test_results['metrics']
            }
            save_sklearn_model(model, model_path, model_info)
            
        else:
            # Train transformer model
            model_name_mapping = {
                'roberta': 'roberta-base',
                'bert': 'bert-base-uncased'
            }
            base_model = model_name_mapping[args.model_type]
            
            model, tokenizer, test_results = train_transformer_model(
                train_data, val_data, test_data, base_model
            )
            
            # Save model
            model_dir = output_dir / model_name
            model_utils = ModelUtils(base_model)
            model_info = {
                'model_type': args.model_type,
                'base_model': base_model,
                'training_params': vars(args),
                'data_stats': stats,
                'test_metrics': test_results['metrics']
            }
            model_utils.save_model_and_tokenizer(model, tokenizer, model_dir, model_info)
        
        # Generate evaluation report
        report_path = output_dir / f"{model_name}_evaluation_report"
        generate_evaluation_report(test_results, str(report_path))
        
        # Log final results
        metrics = test_results['metrics']
        logger.info("Training completed successfully!")
        logger.info(f"Final test metrics:")
        logger.info(f"  R² Score: {metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
        logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        # Save training summary
        summary = {
            'model_name': model_name,
            'model_type': args.model_type,
            'training_completed': datetime.now().isoformat(),
            'final_metrics': metrics,
            'data_stats': stats,
            'config': vars(args)
        }
        
        with open(output_dir / f"{model_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Model saved to: {output_dir / model_name}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
