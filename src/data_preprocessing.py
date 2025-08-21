"""
Data preprocessing utilities for sentiment analysis.
Handles Twitter text cleaning, tokenization, and normalization.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterTextPreprocessor:
    """
    Advanced text preprocessor specifically designed for Twitter sentiment analysis.
    """
    
    def __init__(self, remove_stopwords: bool = False, lowercase: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            remove_stopwords: Whether to remove English stopwords
            lowercase: Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for text cleaning."""
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.rt_pattern = re.compile(r'^RT\s+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.multiple_spaces = re.compile(r'\s+')
        self.repeated_chars = re.compile(r'(.)\1{2,}')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove RT prefix
        text = self.rt_pattern.sub('', text)
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove mentions and hashtags but keep the text
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        
        # Remove emojis (optional - some might carry sentiment)
        # text = self.emoji_pattern.sub(' ', text)
        
        # Replace repeated characters (e.g., "sooooo" -> "soo")
        text = self.repeated_chars.sub(r'\1\1', text)
        
        # Remove numbers (optional - depends on use case)
        # text = self.number_pattern.sub(' ', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = self.multiple_spaces.sub(' ', text)
        text = text.strip()
        
        return text
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and optionally remove stopwords.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        cleaned_text = self.clean_text(text)
        
        if self.remove_stopwords:
            tokens = self.tokenize_and_filter(cleaned_text)
            return ' '.join(tokens)
        
        return cleaned_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess_text(text) for text in texts]


class SentimentDataLoader:
    """
    Data loader for sentiment analysis datasets with score normalization.
    """
    
    def __init__(self, preprocessor: Optional[TwitterTextPreprocessor] = None):
        """
        Initialize the data loader.
        
        Args:
            preprocessor: Text preprocessor instance
        """
        self.preprocessor = preprocessor or TwitterTextPreprocessor()
    
    def load_sentiment140(self, filepath: str) -> pd.DataFrame:
        """
        Load Sentiment140 dataset format.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with 'text' and 'sentiment' columns
        """
        # Sentiment140 format: sentiment,id,date,query,user,text
        column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        
        try:
            df = pd.read_csv(filepath, names=column_names, encoding='latin-1')
            
            # Convert binary sentiment (0,4) to 1-5 scale
            df['sentiment'] = df['sentiment'].map({0: 1, 4: 5})
            
            # Keep only text and sentiment
            df = df[['text', 'sentiment']].copy()
            
            logger.info(f"Loaded {len(df)} samples from Sentiment140 dataset")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Sentiment140 dataset: {e}")
            raise
    
    def load_custom_dataset(self, filepath: str, text_column: str = 'text', 
                          sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """
        Load custom dataset format.
        
        Args:
            filepath: Path to the CSV file
            text_column: Name of the text column
            sentiment_column: Name of the sentiment column
            
        Returns:
            DataFrame with 'text' and 'sentiment' columns
        """
        try:
            df = pd.read_csv(filepath)
            
            # Rename columns to standard format
            df = df[[text_column, sentiment_column]].copy()
            df.columns = ['text', 'sentiment']
            
            logger.info(f"Loaded {len(df)} samples from custom dataset")
            return df
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            raise
    
    def normalize_scores(self, scores: np.ndarray, target_min: float = 1.0, 
                        target_max: float = 5.0) -> np.ndarray:
        """
        Normalize sentiment scores to target range.
        
        Args:
            scores: Original scores array
            target_min: Target minimum value
            target_max: Target maximum value
            
        Returns:
            Normalized scores array
        """
        scores = np.array(scores)
        
        # Current range
        current_min = scores.min()
        current_max = scores.max()
        
        if current_min == current_max:
            # All scores are the same
            return np.full_like(scores, (target_min + target_max) / 2)
        
        # Normalize to [0, 1] then scale to target range
        normalized = (scores - current_min) / (current_max - current_min)
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled
    
    def create_synthetic_scores(self, binary_sentiment: np.ndarray, 
                              add_noise: bool = True) -> np.ndarray:
        """
        Convert binary sentiment to continuous 1-5 scores with realistic distribution.
        
        Args:
            binary_sentiment: Binary sentiment array (0 or 1, negative or positive)
            add_noise: Whether to add realistic noise to scores
            
        Returns:
            Continuous sentiment scores (1-5)
        """
        scores = np.zeros_like(binary_sentiment, dtype=float)
        
        # Negative sentiment: bias towards 1-2
        negative_mask = binary_sentiment == 0
        scores[negative_mask] = np.random.uniform(1.0, 2.5, size=negative_mask.sum())
        
        # Positive sentiment: bias towards 4-5
        positive_mask = binary_sentiment == 1
        scores[positive_mask] = np.random.uniform(3.5, 5.0, size=positive_mask.sum())
        
        if add_noise:
            # Add small amount of realistic noise
            noise = np.random.normal(0, 0.1, size=len(scores))
            scores += noise
            scores = np.clip(scores, 1.0, 5.0)
        
        return scores
    
    def prepare_dataset(self, df: pd.DataFrame, 
                       test_size: float = 0.2, 
                       validation_size: float = 0.1,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare dataset with train/validation/test splits.
        
        Args:
            df: Input DataFrame with 'text' and 'sentiment' columns
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Clean text data
        logger.info("Preprocessing text data...")
        df['text'] = self.preprocessor.preprocess_batch(df['text'].tolist())
        
        # Remove empty texts
        df = df[df['text'].str.len() > 0].reset_index(drop=True)
        
        # Ensure sentiment scores are in 1-5 range
        if df['sentiment'].min() < 1 or df['sentiment'].max() > 5:
            logger.info("Normalizing sentiment scores to 1-5 range...")
            df['sentiment'] = self.normalize_scores(df['sentiment'].values)
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=pd.qcut(df['sentiment'], q=5, duplicates='drop')
        )
        
        # Second split: separate train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=random_state,
            stratify=pd.qcut(train_val_df['sentiment'], q=5, duplicates='drop')
        )
        
        logger.info(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df


def create_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        Sample DataFrame with text and sentiment columns
    """
    np.random.seed(42)
    
    positive_texts = [
        "I love this product! Amazing quality and fast delivery.",
        "Best experience ever! Highly recommend to everyone.",
        "Excellent service and great customer support.",
        "Perfect! Exactly what I was looking for.",
        "Outstanding quality and value for money.",
        "Amazing features and easy to use interface.",
        "Great product, works perfectly as described.",
        "Fantastic quality and quick shipping.",
        "Love it! Better than expected.",
        "Superb quality and excellent design."
    ]
    
    negative_texts = [
        "Terrible product, completely disappointed.",
        "Worst experience ever, would not recommend.",
        "Poor quality and overpriced.",
        "Doesn't work as advertised, waste of money.",
        "Bad customer service and slow delivery.",
        "Cheaply made and breaks easily.",
        "Not worth the price, poor performance.",
        "Disappointed with the quality.",
        "Doesn't meet expectations at all.",
        "Poor design and difficult to use."
    ]
    
    neutral_texts = [
        "It's okay, nothing special but works fine.",
        "Average product, meets basic requirements.",
        "Decent quality for the price point.",
        "Works as expected, no complaints.",
        "Standard product, nothing remarkable.",
        "Good enough for basic needs.",
        "Fair quality and reasonable price.",
        "Acceptable performance overall.",
        "Not bad, but not great either.",
        "Meets minimum requirements."
    ]
    
    # Generate samples
    texts = []
    sentiments = []
    
    for _ in range(n_samples):
        category = np.random.choice(['positive', 'negative', 'neutral'], 
                                  p=[0.4, 0.3, 0.3])
        
        if category == 'positive':
            text = np.random.choice(positive_texts)
            sentiment = np.random.uniform(3.5, 5.0)
        elif category == 'negative':
            text = np.random.choice(negative_texts)
            sentiment = np.random.uniform(1.0, 2.5)
        else:
            text = np.random.choice(neutral_texts)
            sentiment = np.random.uniform(2.5, 3.5)
        
        texts.append(text)
        sentiments.append(sentiment)
    
    return pd.DataFrame({'text': texts, 'sentiment': sentiments})


if __name__ == "__main__":
    # Example usage
    preprocessor = TwitterTextPreprocessor(remove_stopwords=True)
    loader = SentimentDataLoader(preprocessor)
    
    # Create sample dataset
    sample_df = create_sample_dataset(1000)
    
    # Prepare splits
    train_df, val_df, test_df = loader.prepare_dataset(sample_df)
    
    print(f"Dataset created with {len(train_df)} training samples")
    print(f"Sample preprocessed text: {train_df.iloc[0]['text']}")
    print(f"Sample sentiment score: {train_df.iloc[0]['sentiment']:.2f}")
