"""
Testing and evaluation script for sentiment analysis models.
Provides comprehensive evaluation and visualization of trained models.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_processor import DataProcessor
from utils.text_preprocessor import TextPreprocessor
from utils.model_utils import ModelUtils, load_sklearn_model
from utils.evaluation import (
    evaluate_model, calculate_metrics, plot_predictions_vs_actual,
    plot_confusion_matrix, generate_evaluation_report
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path, text_column='review', rating_column='rating'):
    """Load and preprocess test data."""
    logger.info(f"Loading test data from {data_path}")
    
    data_processor = DataProcessor()
    df = data_processor.load_data(data_path)
    
    # Auto-detect columns if defaults don't exist
    if text_column not in df.columns:
        text_column = df.columns[0]
    if rating_column not in df.columns:
        rating_column = df.columns[1]
    
    logger.info(f"Using columns: text='{text_column}', rating='{rating_column}'")
    
    # Preprocess data
    df_processed = data_processor.preprocess_data(
        text_column=text_column,
        rating_column=rating_column
    )
    
    texts = data_processor.processed_data['texts']
    ratings = data_processor.processed_data['ratings']
    
    return texts, ratings

def test_sklearn_model(model_path, test_texts, test_ratings, preprocessor_path=None):
    """Test a scikit-learn model."""
    logger.info(f"Testing sklearn model: {model_path}")
    
    # Load model
    model = load_sklearn_model(model_path)
    
    # Apply text preprocessing if preprocessor is available
    if preprocessor_path and Path(preprocessor_path).exists():
        logger.info("Applying text preprocessing...")
        preprocessor = TextPreprocessor.load_preprocessor(preprocessor_path)
        test_texts = preprocessor.preprocess_batch(test_texts)
    
    # Evaluate model
    test_results = evaluate_model(
        model, test_texts, test_ratings, 
        task_type='regression', model_type='sklearn'
    )
    
    return test_results

def test_transformer_model(model_path, test_texts, test_ratings):
    """Test a transformer model."""
    logger.info(f"Testing transformer model: {model_path}")
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    
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
    
    # Load model and tokenizer
    model_utils = ModelUtils()
    model, tokenizer = model_utils.load_saved_model(model_path)
    
    # Create test dataset
    test_dataset = SentimentDataset(test_texts, test_ratings, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    model.eval()
    predictions = []
    true_labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = outputs.logits.cpu().numpy().flatten()
            batch_labels = labels.cpu().numpy()
            
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predictions, task_type='regression')
    
    test_results = {
        'model_type': 'transformers',
        'task_type': 'regression',
        'test_samples': len(true_labels),
        'metrics': metrics,
        'predictions': {
            'y_true': true_labels.tolist(),
            'y_pred': predictions.tolist()
        }
    }
    
    return test_results

def run_model_comparison(model_configs, test_texts, test_ratings):
    """Run comparison between multiple models."""
    logger.info("Running model comparison...")
    
    comparison_results = {}
    
    for model_name, config in model_configs.items():
        try:
            logger.info(f"Testing model: {model_name}")
            
            if config['type'] == 'sklearn':
                results = test_sklearn_model(
                    config['path'], test_texts, test_ratings,
                    config.get('preprocessor_path')
                )
            elif config['type'] == 'transformer':
                results = test_transformer_model(
                    config['path'], test_texts, test_ratings
                )
            else:
                logger.warning(f"Unknown model type: {config['type']}")
                continue
            
            comparison_results[model_name] = results
            
            # Log key metrics
            metrics = results['metrics']
            logger.info(f"{model_name} results:")
            logger.info(f"  R² Score: {metrics.get('r2', 'N/A'):.4f}")
            logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            continue
    
    return comparison_results

def generate_visualizations(test_results, output_dir, model_name):
    """Generate evaluation visualizations."""
    logger.info("Generating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    y_true = np.array(test_results['predictions']['y_true'])
    y_pred = np.array(test_results['predictions']['y_pred'])
    
    # Predictions vs actual plot
    plot_predictions_vs_actual(
        y_true, y_pred,
        save_path=output_path / f"{model_name}_predictions_vs_actual.png",
        title=f"{model_name} - Predictions vs Actual"
    )
    
    # Confusion matrix (for rounded predictions)
    plot_confusion_matrix(
        y_true, y_pred,
        labels=['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
        save_path=output_path / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} - Confusion Matrix"
    )

def analyze_prediction_errors(test_results, output_dir, model_name, test_texts=None):
    """Analyze prediction errors in detail."""
    logger.info("Analyzing prediction errors...")
    
    y_true = np.array(test_results['predictions']['y_true'])
    y_pred = np.array(test_results['predictions']['y_pred'])
    
    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)
    
    # Create error analysis DataFrame
    error_df = pd.DataFrame({
        'true_rating': y_true,
        'predicted_rating': y_pred,
        'absolute_error': abs_errors,
        'residual': y_true - y_pred
    })
    
    if test_texts:
        error_df['text'] = test_texts[:len(y_true)]  # Ensure same length
    
    # Sort by largest errors
    error_df_sorted = error_df.sort_values('absolute_error', ascending=False)
    
    # Save error analysis
    output_path = Path(output_dir)
    error_df_sorted.to_csv(output_path / f"{model_name}_error_analysis.csv", index=False)
    
    # Generate error statistics
    error_stats = {
        'worst_predictions': error_df_sorted.head(10).to_dict('records'),
        'error_distribution': {
            'errors_0_to_0.5': len(error_df[abs_errors <= 0.5]),
            'errors_0.5_to_1': len(error_df[(abs_errors > 0.5) & (abs_errors <= 1)]),
            'errors_1_to_1.5': len(error_df[(abs_errors > 1) & (abs_errors <= 1.5)]),
            'errors_1.5_to_2': len(error_df[(abs_errors > 1.5) & (abs_errors <= 2)]),
            'errors_above_2': len(error_df[abs_errors > 2])
        },
        'error_by_rating': {
            str(rating): {
                'count': len(error_df[error_df['true_rating'] == rating]),
                'mean_error': float(error_df[error_df['true_rating'] == rating]['absolute_error'].mean()),
                'std_error': float(error_df[error_df['true_rating'] == rating]['absolute_error'].std())
            }
            for rating in [1, 2, 3, 4, 5]
            if len(error_df[error_df['true_rating'] == rating]) > 0
        }
    }
    
    # Save error statistics
    with open(output_path / f"{model_name}_error_stats.json", 'w') as f:
        json.dump(error_stats, f, indent=2)
    
    logger.info(f"Error analysis saved to {output_path}")
    return error_stats

def main():
    """Main testing script."""
    parser = argparse.ArgumentParser(description='Test sentiment analysis model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['sklearn', 'transformer'], help='Type of model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--preprocessor_path', type=str, help='Path to text preprocessor (sklearn only)')
    parser.add_argument('--compare_models', type=str, help='JSON file with model comparison configuration')
    parser.add_argument('--generate_visualizations', action='store_true', help='Generate visualization plots')
    parser.add_argument('--analyze_errors', action='store_true', help='Perform detailed error analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting model testing with configuration: {vars(args)}")
    
    try:
        # Load test data
        test_texts, test_ratings = load_test_data(args.test_data)
        logger.info(f"Loaded {len(test_texts)} test samples")
        
        if args.compare_models:
            # Model comparison mode
            with open(args.compare_models, 'r') as f:
                model_configs = json.load(f)
            
            comparison_results = run_model_comparison(model_configs, test_texts, test_ratings)
            
            # Save comparison results
            with open(output_dir / 'model_comparison.json', 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            # Generate comparison report
            logger.info("Model Comparison Results:")
            for model_name, results in comparison_results.items():
                metrics = results['metrics']
                logger.info(f"\n{model_name}:")
                logger.info(f"  R² Score: {metrics.get('r2', 'N/A'):.4f}")
                logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
                logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        else:
            # Single model testing
            model_name = Path(args.model_path).stem
            
            if args.model_type == 'sklearn':
                test_results = test_sklearn_model(
                    args.model_path, test_texts, test_ratings, args.preprocessor_path
                )
            elif args.model_type == 'transformer':
                test_results = test_transformer_model(
                    args.model_path, test_texts, test_ratings
                )
            
            # Generate evaluation report
            report_path = output_dir / f"{model_name}_test_report"
            generate_evaluation_report(test_results, str(report_path))
            
            # Generate visualizations if requested
            if args.generate_visualizations:
                generate_visualizations(test_results, output_dir, model_name)
            
            # Analyze errors if requested
            if args.analyze_errors:
                analyze_prediction_errors(test_results, output_dir, model_name, test_texts)
            
            # Log results
            metrics = test_results['metrics']
            logger.info("Testing completed successfully!")
            logger.info(f"Test results for {model_name}:")
            logger.info(f"  R² Score: {metrics.get('r2', 'N/A'):.4f}")
            logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
