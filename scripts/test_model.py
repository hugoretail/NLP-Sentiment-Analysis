"""
Testing and evaluation script for sentiment analysis models.
Provides comprehensive evaluation and visualization of trained models.
"""
import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.inference import SentimentInference
from utils.text_preprocessor import TextPreprocessor
from utils.model_utils import SentimentModel, ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTester:
    """Comprehensive model testing and evaluation."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.inference = SentimentInference(model_path)
        self.preprocessor = TextPreprocessor()
        self.results = {}
        
    def run_comprehensive_test(self):
        """Run comprehensive model testing."""
        print("🧪 Starting Comprehensive Model Testing")
        print("=" * 50)
        
        # Basic functionality tests
        self._test_basic_functionality()
        
        # Performance tests
        self._test_performance()
        
        # Edge case tests
        self._test_edge_cases()
        
        # Consistency tests
        self._test_consistency()
        
        # Generate report
        self._generate_report()
        
    def _test_basic_functionality(self):
        """Test basic model functionality."""
        print("\n📊 Testing Basic Functionality...")
        
        test_cases = [
            ("I love this product! It's amazing!", "positive"),
            ("This is terrible, I hate it completely.", "negative"),
            ("It's okay, nothing special but adequate.", "neutral"),
            ("Absolutely fantastic experience, highly recommend!", "positive"),
            ("Worst purchase ever, complete waste of money.", "negative")
        ]
        
        results = []
        for text, expected_sentiment in test_cases:
            result = self.inference.predict_single(text)
            results.append({
                'text': text,
                'expected': expected_sentiment,
                'predicted_score': result['score'],
                'predicted_label': result['sentiment_label'],
                'confidence': result['confidence']
            })
            
            print(f"Text: '{text[:40]}{'...' if len(text) > 40 else ''}'")
            print(f"  Expected: {expected_sentiment}")
            print(f"  Predicted: {result['sentiment_label']} (score: {result['score']:.2f})")
            print(f"  Confidence: {result['confidence']:.3f}")
            print()
            
        self.results['basic_functionality'] = results
        
    def _test_performance(self):
        """Test model performance and speed."""
        print("\n⚡ Testing Performance...")
        
        # Single prediction speed
        test_text = "This is a performance test for the sentiment analysis model."
        
        times = []
        for _ in range(10):
            start_time = time.time()
            self.inference.predict_single(test_text)
            times.append(time.time() - start_time)
            
        avg_time = sum(times) / len(times)
        print(f"Average single prediction time: {avg_time:.3f}s")
        
        # Batch prediction speed
        batch_texts = [f"Test text number {i} for batch processing." for i in range(50)]
        
        start_time = time.time()
        batch_results = self.inference.predict_batch(batch_texts)
        batch_time = time.time() - start_time
        
        print(f"Batch prediction time (50 texts): {batch_time:.3f}s")
        print(f"Average per text in batch: {batch_time/len(batch_texts):.3f}s")
        
        self.results['performance'] = {
            'single_prediction_time': avg_time,
            'batch_prediction_time': batch_time,
            'batch_size': len(batch_texts),
            'avg_batch_per_text': batch_time/len(batch_texts)
        }
        
    def _test_edge_cases(self):
        """Test model with edge cases."""
        print("\n🔍 Testing Edge Cases...")
        
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "a",  # Single character
            "OK",  # Very short
            "!" * 50,  # Repeated punctuation
            "😀😀😀😀😀",  # Only emojis
            "https://example.com",  # Only URL
            "@user #hashtag",  # Only mentions and hashtags
            "SHOUTING IN ALL CAPS!!!",  # All caps
            "Can't won't shouldn't wouldn't",  # Contractions
            "This is a very " + "long " * 30 + "text.",  # Very long text
        ]
        
        edge_results = []
        for text in edge_cases:
            try:
                result = self.inference.predict_single(text)
                edge_results.append({
                    'text': repr(text),
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'label': result['sentiment_label'],
                    'success': True
                })
                print(f"✅ {repr(text)[:30]}... -> {result['sentiment_label']} ({result['score']:.2f})")
            except Exception as e:
                edge_results.append({
                    'text': repr(text),
                    'error': str(e),
                    'success': False
                })
                print(f"❌ {repr(text)[:30]}... -> ERROR: {e}")
                
        self.results['edge_cases'] = edge_results
        
    def _test_consistency(self):
        """Test model consistency."""
        print("\n🔄 Testing Consistency...")
        
        test_texts = [
            "This product is great!",
            "I hate this service.",
            "Average quality, nothing special."
        ]
        
        consistency_results = []
        for text in test_texts:
            predictions = []
            for _ in range(5):  # Run same text 5 times
                result = self.inference.predict_single(text)
                predictions.append(result['score'])
                
            std_dev = np.std(predictions) if len(predictions) > 1 else 0
            avg_score = np.mean(predictions)
            
            consistency_results.append({
                'text': text,
                'predictions': predictions,
                'average_score': avg_score,
                'std_deviation': std_dev,
                'consistent': std_dev < 0.1  # Consider consistent if std < 0.1
            })
            
            print(f"Text: '{text}'")
            print(f"  Predictions: {[f'{p:.2f}' for p in predictions]}")
            print(f"  Average: {avg_score:.2f}, Std Dev: {std_dev:.3f}")
            print(f"  Consistent: {'✅' if std_dev < 0.1 else '❌'}")
            print()
            
        self.results['consistency'] = consistency_results
        
    def _generate_report(self):
        """Generate comprehensive test report."""
        print("\n📋 Test Report")
        print("=" * 50)
        
        # Basic functionality summary
        basic_results = self.results['basic_functionality']
        print(f"\n🔧 Basic Functionality: {len(basic_results)} tests")
        
        # Performance summary
        perf = self.results['performance']
        print(f"\n⚡ Performance:")
        print(f"  Single prediction: {perf['single_prediction_time']:.3f}s")
        print(f"  Batch processing: {perf['avg_batch_per_text']:.3f}s per text")
        
        # Edge cases summary
        edge_results = self.results['edge_cases']
        successful_edges = sum(1 for r in edge_results if r['success'])
        print(f"\n🔍 Edge Cases: {successful_edges}/{len(edge_results)} passed")
        
        # Consistency summary
        consistency_results = self.results['consistency']
        consistent_count = sum(1 for r in consistency_results if r['consistent'])
        print(f"\n🔄 Consistency: {consistent_count}/{len(consistency_results)} consistent")
        
        # Overall assessment
        print(f"\n📊 Overall Assessment:")
        
        performance_score = 100 if perf['single_prediction_time'] < 1.0 else 50
        edge_score = (successful_edges / len(edge_results)) * 100
        consistency_score = (consistent_count / len(consistency_results)) * 100
        
        overall_score = (performance_score + edge_score + consistency_score) / 3
        
        print(f"  Performance Score: {performance_score:.1f}%")
        print(f"  Edge Case Score: {edge_score:.1f}%")
        print(f"  Consistency Score: {consistency_score:.1f}%")
        print(f"  Overall Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("🎉 Excellent! Model is ready for production.")
        elif overall_score >= 60:
            print("✅ Good! Model performs well with minor issues.")
        else:
            print("⚠️  Needs improvement. Consider model retraining.")
            
    def export_results(self, filename: str = "model_test_results.json"):
        """Export test results to file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        # Deep convert all numpy types
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(v) for v in data]
            else:
                return convert_numpy(data)
                
        converted_results = deep_convert(self.results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 Results exported to {filename}")

def run_quick_test():
    """Run a quick functionality test."""
    print("💨 Quick Model Test")
    print("-" * 30)
    
    tester = ModelTester()
    
    # Quick test cases
    quick_tests = [
        "Great product!",
        "Terrible quality.",
        "It's okay."
    ]
    
    for text in quick_tests:
        result = tester.inference.predict_single(text)
        print(f"'{text}' -> {result['sentiment_label']} ({result['score']:.2f})")
        
    print("\n✅ Quick test completed!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test sentiment analysis model')
    parser.add_argument('--model', type=str, help='Path to model directory')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--export', type=str, help='Export results to file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    if args.quick:
        run_quick_test()
        return
        
    # Run comprehensive test
    tester = ModelTester(args.model)
    tester.run_comprehensive_test()
    
    # Export results if requested
    if args.export:
        tester.export_results(args.export)

if __name__ == "__main__":
    # Import numpy here to avoid import errors in minimal environments
    try:
        import numpy as np
    except ImportError:
        print("Warning: NumPy not available, some features may be limited.")
        # Create mock numpy for basic functionality
        class MockNumPy:
            def std(self, data): return 0.0
            def mean(self, data): return sum(data) / len(data) if data else 0.0
        np = MockNumPy()
    
    main()
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
