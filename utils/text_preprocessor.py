"""
Text preprocessing utilities for sentiment analysis.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import pickle

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utilities for sentiment analysis."""
    
    def __init__(self, 
                 language: str = 'english',
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 min_word_length: int = 2):
        """
        Initialize TextPreprocessor.
        
        Args:
            language: Language for stopwords
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            min_word_length: Minimum word length to keep
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        # Add custom stopwords for reviews
        custom_stopwords = {
            'would', 'could', 'should', 'might', 'may', 'will', 'shall',
            'one', 'two', 'three', 'first', 'second', 'third',
            'really', 'quite', 'very', 'much', 'many', 'well'
        }
        self.stop_words.update(custom_stopwords)
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub('', text)
    
    def clean_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def clean_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = re.compile(r'\S+@\S+')
        return email_pattern.sub('', text)
    
    def clean_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,15}')
        return phone_pattern.sub('', text)
    
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace from text."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_special_characters(self, text: str, keep_chars: str = '') -> str:
        """Remove special characters except those specified."""
        if self.remove_punctuation:
            # Keep only alphanumeric and specified characters
            pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
        else:
            # Keep alphanumeric, basic punctuation, and specified characters
            pattern = f'[^a-zA-Z0-9\\s\\.\\!\\?\\,\\;\\:\\-\\(\\){re.escape(keep_chars)}]'
        
        return re.sub(pattern, '', text)
    
    def remove_numbers_func(self, text: str) -> str:
        """Remove numbers from text."""
        if self.remove_numbers:
            return re.sub(r'\d+', '', text)
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if NLTK fails
            tokens = text.split()
        
        return tokens
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter tokens based on preprocessing settings."""
        filtered_tokens = []
        
        for token in tokens:
            # Skip empty tokens
            if not token:
                continue
            
            # Convert to lowercase
            if self.lowercase:
                token = token.lower()
            
            # Skip short words
            if len(token) < self.min_word_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Skip if only punctuation
            if token in string.punctuation:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        if not self.lemmatize or not self.lemmatizer:
            return tokens
        
        lemmatized = []
        for token in tokens:
            try:
                lemmatized.append(self.lemmatizer.lemmatize(token))
            except:
                lemmatized.append(token)
        
        return lemmatized
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Basic cleaning
        text = self.clean_html(text)
        text = self.clean_urls(text)
        text = self.clean_emails(text)
        text = self.clean_phone_numbers(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Remove numbers if requested
        text = self.remove_numbers_func(text)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        logger.info(f"Preprocessing {len(texts)} texts...")
        
        processed_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
            
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        logger.info(f"Preprocessing complete. Processed {len(processed_texts)} texts")
        return processed_texts
    
    def get_word_frequencies(self, texts: List[str], top_n: int = 100) -> Dict[str, int]:
        """
        Get word frequencies from a list of texts.
        
        Args:
            texts: List of preprocessed texts
            top_n: Number of top words to return
            
        Returns:
            Dictionary of word frequencies
        """
        from collections import Counter
        
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        return dict(word_freq.most_common(top_n))
    
    def save_preprocessor(self, file_path: str) -> None:
        """Save preprocessor configuration."""
        config = {
            'language': self.language,
            'remove_stopwords': self.remove_stopwords,
            'lemmatize': self.lemmatize,
            'lowercase': self.lowercase,
            'remove_punctuation': self.remove_punctuation,
            'remove_numbers': self.remove_numbers,
            'min_word_length': self.min_word_length,
            'stop_words': list(self.stop_words)
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load_preprocessor(cls, file_path: str) -> 'TextPreprocessor':
        """Load preprocessor configuration."""
        with open(file_path, 'rb') as f:
            config = pickle.load(f)
        
        preprocessor = cls(
            language=config['language'],
            remove_stopwords=config['remove_stopwords'],
            lemmatize=config['lemmatize'],
            lowercase=config['lowercase'],
            remove_punctuation=config['remove_punctuation'],
            remove_numbers=config['remove_numbers'],
            min_word_length=config['min_word_length']
        )
        
        preprocessor.stop_words = set(config['stop_words'])
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor
    
    def get_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about text data.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary containing text statistics
        """
        import numpy as np
        
        if not texts:
            return {}
        
        # Calculate basic statistics
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        # Get word frequencies
        word_freq = self.get_word_frequencies(texts)
        
        stats = {
            'total_texts': len(texts),
            'word_count_stats': {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'std': np.std(lengths)
            },
            'char_count_stats': {
                'mean': np.mean(char_lengths),
                'median': np.median(char_lengths),
                'min': np.min(char_lengths),
                'max': np.max(char_lengths),
                'std': np.std(char_lengths)
            },
            'vocabulary_size': len(word_freq),
            'top_words': dict(list(word_freq.items())[:20])
        }
        
        return stats
