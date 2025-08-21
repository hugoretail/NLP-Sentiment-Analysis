"""
Flask application for sentiment analysis with advanced visualizations.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import SentimentInferenceEngine, get_inference_engine
from src.data_preprocessing import TwitterTextPreprocessor
from src.evaluation import SentimentEvaluationReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'sentiment-analysis-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
inference_engine = None
analysis_history = []

# Model path (relative to app directory)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')

def initialize_model():
    """Initialize the sentiment analysis model."""
    global inference_engine
    try:
        inference_engine = get_inference_engine(MODEL_PATH)
        logger.info("Sentiment analysis model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

def startup():
    """Initialize the application on startup."""
    if not initialize_model():
        logger.error("Failed to initialize model. Some endpoints may not work.")

# Initialize on app startup (Flask 2.3+ compatible)
with app.app_context():
    startup()

# ==================== API ENDPOINTS ====================

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    model_status = inference_engine is not None
    return jsonify({
        'status': 'healthy' if model_status else 'degraded',
        'model_loaded': model_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    """
    Predict sentiment for a single text.
    
    Expected JSON: {"text": "text to analyze", "detailed": true/false}
    """
    try:
        if not inference_engine:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        detailed = data.get('detailed', True)
        
        if not text or not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get prediction
        if detailed:
            result = inference_engine.analyze_text_sentiment(text)
        else:
            result = inference_engine.predict_sentiment(text)
        
        # Store in history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text[:100] + '...' if len(text) > 100 else text,
            'sentiment_score': result.get('sentiment_score', 3.0),
            'prediction_class': result.get('prediction_class', 'neutral'),
            'confidence': result.get('confidence', 0.0)
        }
        analysis_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(analysis_history) > 100:
            analysis_history.pop(0)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict sentiment for multiple texts.
    
    Expected JSON: {"texts": ["text1", "text2", ...], "batch_size": 16}
    """
    try:
        if not inference_engine:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
        
        texts = data['texts']
        batch_size = data.get('batch_size', 16)
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        if len(texts) == 0:
            return jsonify({'error': 'Empty texts list'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts allowed per batch'}), 400
        
        # Get predictions
        results = inference_engine.predict_batch(texts, batch_size)
        
        # Add to history (summarized)
        avg_score = np.mean([r.get('sentiment_score', 3.0) for r in results])
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': f'Batch analysis ({len(texts)} texts)',
            'sentiment_score': round(avg_score, 2),
            'prediction_class': 'batch',
            'confidence': np.mean([r.get('confidence', 0.0) for r in results])
        }
        analysis_history.append(history_entry)
        
        # Calculate batch statistics
        scores = [r.get('sentiment_score', 3.0) for r in results]
        batch_stats = {
            'count': len(results),
            'mean_score': round(np.mean(scores), 3),
            'std_score': round(np.std(scores), 3),
            'min_score': round(np.min(scores), 3),
            'max_score': round(np.max(scores), 3),
            'class_distribution': {
                'positive': sum(1 for r in results if r.get('prediction_class') == 'positive'),
                'neutral': sum(1 for r in results if r.get('prediction_class') == 'neutral'),
                'negative': sum(1 for r in results if r.get('prediction_class') == 'negative')
            }
        }
        
        return jsonify({
            'results': results,
            'batch_statistics': batch_stats
        })
        
    except Exception as e:
        logger.error(f"Error in batch_predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def get_model_info():
    """Get comprehensive model information."""
    try:
        if not inference_engine:
            return jsonify({'error': 'Model not loaded'}), 500
        
        model_info = inference_engine.get_model_info()
        performance_stats = inference_engine.get_performance_stats()
        
        return jsonify({
            'model_info': model_info,
            'performance_stats': performance_stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_model_info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_analysis_history():
    """Get analysis history."""
    try:
        # Get query parameters
        limit = request.args.get('limit', default=50, type=int)
        offset = request.args.get('offset', default=0, type=int)
        
        # Apply pagination
        total_count = len(analysis_history)
        start_idx = max(0, total_count - offset - limit)
        end_idx = total_count - offset
        
        paginated_history = analysis_history[start_idx:end_idx]
        paginated_history.reverse()  # Most recent first
        
        return jsonify({
            'history': paginated_history,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error in get_analysis_history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_analysis_history():
    """Clear analysis history."""
    try:
        global analysis_history
        analysis_history.clear()
        
        return jsonify({'message': 'History cleared successfully'})
        
    except Exception as e:
        logger.error(f"Error in clear_analysis_history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    """Get analytics and insights from analysis history."""
    try:
        if len(analysis_history) == 0:
            return jsonify({
                'message': 'No analysis history available',
                'analytics': None
            })
        
        # Calculate analytics
        scores = [entry['sentiment_score'] for entry in analysis_history if entry['prediction_class'] != 'batch']
        classes = [entry['prediction_class'] for entry in analysis_history if entry['prediction_class'] != 'batch']
        confidences = [entry['confidence'] for entry in analysis_history if entry['prediction_class'] != 'batch']
        
        if len(scores) == 0:
            return jsonify({
                'message': 'No individual analyses in history',
                'analytics': None
            })
        
        # Score statistics
        score_stats = {
            'mean': round(np.mean(scores), 3),
            'std': round(np.std(scores), 3),
            'min': round(np.min(scores), 3),
            'max': round(np.max(scores), 3),
            'median': round(np.median(scores), 3)
        }
        
        # Class distribution
        class_counts = {
            'positive': classes.count('positive'),
            'neutral': classes.count('neutral'),
            'negative': classes.count('negative')
        }
        
        # Confidence statistics
        confidence_stats = {
            'mean': round(np.mean(confidences), 3),
            'std': round(np.std(confidences), 3),
            'min': round(np.min(confidences), 3),
            'max': round(np.max(confidences), 3)
        }
        
        # Time-based analysis (last 24 hours by hour)
        from datetime import datetime, timedelta
        now = datetime.now()
        hourly_counts = {i: 0 for i in range(24)}
        hourly_avg_scores = {i: [] for i in range(24)}
        
        for entry in analysis_history:
            if entry['prediction_class'] != 'batch':
                entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
                hours_ago = int((now - entry_time).total_seconds() / 3600)
                if hours_ago < 24:
                    hour_key = 23 - hours_ago
                    hourly_counts[hour_key] += 1
                    hourly_avg_scores[hour_key].append(entry['sentiment_score'])
        
        # Calculate hourly averages
        hourly_avg = {}
        for hour, scores_list in hourly_avg_scores.items():
            hourly_avg[hour] = round(np.mean(scores_list), 2) if scores_list else 0
        
        analytics = {
            'total_analyses': len(scores),
            'score_statistics': score_stats,
            'class_distribution': class_counts,
            'confidence_statistics': confidence_stats,
            'hourly_analysis': {
                'counts': hourly_counts,
                'average_scores': hourly_avg
            },
            'insights': generate_insights(score_stats, class_counts, confidence_stats)
        }
        
        return jsonify({'analytics': analytics})
        
    except Exception as e:
        logger.error(f"Error in get_analytics: {e}")
        return jsonify({'error': str(e)}), 500

def generate_insights(score_stats: Dict, class_counts: Dict, confidence_stats: Dict) -> List[str]:
    """Generate insights from analytics data."""
    insights = []
    
    # Score insights
    if score_stats['mean'] > 4.0:
        insights.append("Overall sentiment is very positive!")
    elif score_stats['mean'] > 3.5:
        insights.append("Overall sentiment leans positive.")
    elif score_stats['mean'] < 2.0:
        insights.append("Overall sentiment is quite negative.")
    elif score_stats['mean'] < 2.5:
        insights.append("Overall sentiment leans negative.")
    else:
        insights.append("Overall sentiment is neutral.")
    
    # Class distribution insights
    total_analyses = sum(class_counts.values())
    if total_analyses > 0:
        pos_ratio = class_counts['positive'] / total_analyses
        neg_ratio = class_counts['negative'] / total_analyses
        
        if pos_ratio > 0.6:
            insights.append(f"{pos_ratio*100:.0f}% of analyses were positive.")
        elif neg_ratio > 0.6:
            insights.append(f"{neg_ratio*100:.0f}% of analyses were negative.")
        elif class_counts['neutral'] / total_analyses > 0.5:
            insights.append("Most analyses were classified as neutral.")
    
    # Confidence insights
    if confidence_stats['mean'] > 0.8:
        insights.append("Model predictions have very high confidence.")
    elif confidence_stats['mean'] > 0.6:
        insights.append("Model predictions have good confidence.")
    elif confidence_stats['mean'] < 0.4:
        insights.append("Model predictions have low confidence - results may be uncertain.")
    
    # Variability insights
    if score_stats['std'] > 1.5:
        insights.append("High variability in sentiment scores - diverse range of content analyzed.")
    elif score_stats['std'] < 0.5:
        insights.append("Low variability in sentiment scores - content is fairly consistent.")
    
    return insights

@app.route('/api/compare', methods=['POST'])
def compare_texts():
    """
    Compare sentiment of multiple texts side by side.
    
    Expected JSON: {"texts": ["text1", "text2", ...], "labels": ["label1", "label2", ...]}
    """
    try:
        if not inference_engine:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
        
        texts = data['texts']
        labels = data.get('labels', [f"Text {i+1}" for i in range(len(texts))])
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        if len(texts) > 10:
            return jsonify({'error': 'Maximum 10 texts allowed for comparison'}), 400
        
        # Ensure labels list has same length as texts
        if len(labels) < len(texts):
            labels.extend([f"Text {i+1}" for i in range(len(labels), len(texts))])
        
        # Get detailed predictions
        results = []
        for i, text in enumerate(texts):
            prediction = inference_engine.analyze_text_sentiment(text)
            results.append({
                'label': labels[i],
                'text': text,
                'prediction': prediction
            })
        
        # Calculate comparison statistics
        scores = [r['prediction']['sentiment_score'] for r in results]
        comparison_stats = {
            'highest_score': {
                'value': max(scores),
                'label': labels[scores.index(max(scores))]
            },
            'lowest_score': {
                'value': min(scores),
                'label': labels[scores.index(min(scores))]
            },
            'score_range': round(max(scores) - min(scores), 3),
            'average_score': round(np.mean(scores), 3),
            'most_confident': {
                'value': max(r['prediction']['confidence'] for r in results),
                'label': labels[np.argmax([r['prediction']['confidence'] for r in results])]
            }
        }
        
        return jsonify({
            'results': results,
            'comparison_statistics': comparison_stats
        })
        
    except Exception as e:
        logger.error(f"Error in compare_texts: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== STATIC FILE SERVING ====================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors."""
    return jsonify({'error': 'File too large'}), 413

# ==================== DEVELOPMENT ROUTES ====================

@app.route('/api/debug/cache_stats')
def get_cache_stats():
    """Get cache statistics (debug endpoint)."""
    if not inference_engine:
        return jsonify({'error': 'Model not loaded'}), 500
    
    stats = inference_engine.get_performance_stats()
    return jsonify(stats)

@app.route('/api/debug/clear_cache', methods=['POST'])
def clear_cache():
    """Clear inference cache (debug endpoint)."""
    if not inference_engine:
        return jsonify({'error': 'Model not loaded'}), 500
    
    inference_engine.clear_cache()
    return jsonify({'message': 'Cache cleared successfully'})

# ==================== MAIN APPLICATION ====================

if __name__ == '__main__':
    # Ensure model directory exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model directory not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Initialize the model
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        sys.exit(1)
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
