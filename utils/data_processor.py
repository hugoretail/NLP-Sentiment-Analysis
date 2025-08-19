"""
Data processing utilities for sentiment analysis.
Handles data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import re
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for sentiment analysis."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path) if data_path else None
        self.df = None
        self.processed_data = {}
        
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.json', '.jsonl']:
                df = pd.read_json(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            logger.info(f"Loaded data with shape: {df.shape}")
            self.df = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean text data.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def validate_ratings(self, ratings: pd.Series) -> pd.Series:
        """
        Validate and clean rating values.
        
        Args:
            ratings: Series containing rating values
            
        Returns:
            Cleaned ratings series
        """
        # Convert to numeric
        ratings = pd.to_numeric(ratings, errors='coerce')
        
        # Ensure ratings are between 1 and 5
        ratings = ratings.clip(1, 5)
        
        # Round to nearest integer
        ratings = ratings.round().astype('Int64')
        
        return ratings
    
    def preprocess_data(self, 
                       text_column: str = 'review', 
                       rating_column: str = 'rating',
                       clean_text_flag: bool = True,
                       validate_ratings_flag: bool = True) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            text_column: Name of the text column
            rating_column: Name of the rating column
            clean_text_flag: Whether to clean text
            validate_ratings_flag: Whether to validate ratings
            
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.df.copy()
        
        # Handle missing values
        df = df.dropna(subset=[text_column, rating_column])
        
        # Clean text if requested
        if clean_text_flag:
            logger.info("Cleaning text data...")
            df[text_column] = df[text_column].apply(self.clean_text)
        
        # Validate ratings if requested
        if validate_ratings_flag:
            logger.info("Validating ratings...")
            df[rating_column] = self.validate_ratings(df[rating_column])
        
        # Remove empty texts
        df = df[df[text_column].str.len() > 0]
        
        # Remove invalid ratings
        df = df.dropna(subset=[rating_column])
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        
        self.processed_data = {
            'texts': df[text_column].tolist(),
            'ratings': df[rating_column].tolist(),
            'dataframe': df
        }
        
        return df
    
    def split_data(self, 
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42,
                   stratify: bool = True) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by ratings
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        if not self.processed_data:
            raise ValueError("No processed data. Call preprocess_data() first.")
        
        from sklearn.model_selection import train_test_split
        
        texts = self.processed_data['texts']
        ratings = self.processed_data['ratings']
        
        stratify_data = ratings if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, ratings, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_data
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = [], []
        
        train_data = {'texts': X_train, 'ratings': y_train}
        val_data = {'texts': X_val, 'ratings': y_val}
        test_data = {'texts': X_test, 'ratings': y_test}
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return train_data, val_data, test_data
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed data.
        
        Returns:
            Dictionary containing data statistics
        """
        if not self.processed_data:
            raise ValueError("No processed data. Call preprocess_data() first.")
        
        df = self.processed_data['dataframe']
        texts = self.processed_data['texts']
        ratings = self.processed_data['ratings']
        
        # Text statistics
        text_lengths = [len(text.split()) for text in texts]
        
        # Rating distribution
        rating_counts = pd.Series(ratings).value_counts().sort_index()
        
        stats = {
            'total_samples': len(texts),
            'rating_distribution': rating_counts.to_dict(),
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'median': np.median(text_lengths),
                'min': np.min(text_lengths),
                'max': np.max(text_lengths),
                'std': np.std(text_lengths)
            },
            'rating_stats': {
                'mean': np.mean(ratings),
                'median': np.median(ratings),
                'std': np.std(ratings)
            }
        }
        
        return stats
    
    def save_processed_data(self, output_dir: str) -> None:
        """
        Save processed data to files.
        
        Args:
            output_dir: Directory to save processed data
        """
        if not self.processed_data:
            raise ValueError("No processed data to save.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        df = self.processed_data['dataframe']
        df.to_csv(output_path / 'processed_data.csv', index=False)
        
        # Save statistics
        stats = self.get_data_statistics()
        with open(output_path / 'data_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {output_path}")
    
    def load_processed_data(self, input_dir: str) -> None:
        """
        Load previously processed data.
        
        Args:
            input_dir: Directory containing processed data
        """
        input_path = Path(input_dir)
        
        # Load DataFrame
        df_path = input_path / 'processed_data.csv'
        if df_path.exists():
            df = pd.read_csv(df_path)
            self.processed_data = {
                'texts': df.iloc[:, 0].tolist(),  # Assuming first column is text
                'ratings': df.iloc[:, 1].tolist(),  # Assuming second column is rating
                'dataframe': df
            }
            logger.info(f"Loaded processed data with {len(self.processed_data['texts'])} samples")
        else:
            raise FileNotFoundError(f"Processed data file not found: {df_path}")
