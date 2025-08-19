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
    
    """
Text preprocessing utilities for sentiment analysis.
"""
import re
import string
import unicodedata
import logging
from typing import List, Dict, Any, Optional
import contractions
import spacy
from collections import Counter

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Comprehensive text preprocessing for sentiment analysis."""
    
    def __init__(self, load_spacy: bool = False):
        self.load_spacy = load_spacy
        self.nlp = None
        if load_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                
        # Emoji patterns for sentiment
        self.positive_emojis = {
            '😀', '😁', '😂', '🤣', '😃', '😄', '😆', '😊', '😋', '😎', '😍', 
            '🥰', '😘', '🤗', '🤩', '🥳', '😌', '😉', '🙂', '🙃', '😇', '🥺',
            '👍', '👌', '👏', '🙌', '💪', '✨', '⭐', '🌟', '💫', '🎉', '🎊',
            '🔥', '💯', '❤️', '💚', '💙', '💜', '🧡', '💛', '💖', '💕'
        }
        
        self.negative_emojis = {
            '😞', '😔', '😟', '😕', '🙁', '☹️', '😣', '😖', '😫', '😩', '🥺',
            '😢', '😭', '😤', '😠', '😡', '🤬', '😨', '😰', '😱', '🥶', '😵',
            '🤢', '🤮', '😷', '🤒', '🤕', '😈', '👿', '💀', '☠️', '💩',
            '👎', '👊', '🖕', '💔', '⚡', '💥', '🌩️', '❌', '🚫', '⛔'
        }
        
    def preprocess_text(self, text: str, options: Dict[str, bool] = None) -> str:
        """
        Comprehensive text preprocessing.
        
        Args:
            text: Input text to preprocess
            options: Dictionary of preprocessing options
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        if options is None:
            options = {
                'lowercase': True,
                'remove_urls': True,
                'remove_mentions': True,
                'remove_hashtags': False,
                'expand_contractions': True,
                'remove_punctuation': False,
                'remove_extra_whitespace': True,
                'handle_emojis': True,
                'remove_special_chars': False,
                'normalize_unicode': True
            }
            
        # Make a copy
        processed_text = text
        
        # Normalize unicode
        if options.get('normalize_unicode', True):
            processed_text = self.normalize_unicode(processed_text)
            
        # Handle emojis before other processing
        if options.get('handle_emojis', True):
            processed_text = self.handle_emojis(processed_text)
            
        # Remove URLs
        if options.get('remove_urls', True):
            processed_text = self.remove_urls(processed_text)
            
        # Remove mentions
        if options.get('remove_mentions', True):
            processed_text = self.remove_mentions(processed_text)
            
        # Remove hashtags (but keep the text)
        if options.get('remove_hashtags', False):
            processed_text = self.remove_hashtags(processed_text)
            
        # Expand contractions
        if options.get('expand_contractions', True):
            processed_text = self.expand_contractions(processed_text)
            
        # Remove special characters
        if options.get('remove_special_chars', False):
            processed_text = self.remove_special_chars(processed_text)
            
        # Remove punctuation
        if options.get('remove_punctuation', False):
            processed_text = self.remove_punctuation(processed_text)
            
        # Convert to lowercase
        if options.get('lowercase', True):
            processed_text = processed_text.lower()
            
        # Remove extra whitespace
        if options.get('remove_extra_whitespace', True):
            processed_text = self.remove_extra_whitespace(processed_text)
            
        return processed_text.strip()
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKD', text)
        
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        # Also remove www links
        www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(www_pattern, '', text)
        return text
        
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return re.sub(r'@\w+', '', text)
        
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags but keep the text."""
        return re.sub(r'#(\w+)', r'\1', text)
        
    def expand_contractions(self, text: str) -> str:
        """Expand contractions in text."""
        try:
            return contractions.fix(text)
        except Exception:
            # Fallback manual expansion for common contractions
            contractions_dict = {
                "won't": "will not",
                "can't": "cannot",
                "n't": " not",
                "'re": " are",
                "'ve": " have",
                "'ll": " will",
                "'d": " would",
                "'m": " am"
            }
            
            for contraction, expansion in contractions_dict.items():
                text = text.replace(contraction, expansion)
            return text
            
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
        
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing."""
        return re.sub(r'\s+', ' ', text)
        
    def handle_emojis(self, text: str) -> str:
        """Convert emojis to sentiment indicators."""
        # Count sentiment emojis
        positive_count = sum(1 for char in text if char in self.positive_emojis)
        negative_count = sum(1 for char in text if char in self.negative_emojis)
        
        # Remove all emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        
        # Add sentiment indicators based on emoji counts
        if positive_count > negative_count:
            if positive_count >= 3:
                text += " very positive sentiment"
            else:
                text += " positive sentiment"
        elif negative_count > positive_count:
            if negative_count >= 3:
                text += " very negative sentiment"
            else:
                text += " negative sentiment"
                
        return text
        
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract various text features for analysis."""
        features = {}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        features['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        
        # Sentiment indicators
        features['positive_words'] = self._count_positive_words(text)
        features['negative_words'] = self._count_negative_words(text)
        
        # Emoji features
        features['positive_emojis'] = sum(1 for char in text if char in self.positive_emojis)
        features['negative_emojis'] = sum(1 for char in text if char in self.negative_emojis)
        
        return features
        
    def _count_positive_words(self, text: str) -> int:
        """Count positive sentiment words."""
        positive_words = {
            'love', 'like', 'good', 'great', 'excellent', 'amazing', 'awesome',
            'wonderful', 'fantastic', 'perfect', 'best', 'beautiful', 'nice',
            'happy', 'joy', 'pleased', 'satisfied', 'delighted', 'thrilled'
        }
        text_lower = text.lower()
        return sum(1 for word in positive_words if word in text_lower)
        
    def _count_negative_words(self, text: str) -> int:
        """Count negative sentiment words."""
        negative_words = {
            'hate', 'dislike', 'bad', 'terrible', 'awful', 'horrible', 'worst',
            'disgusting', 'annoying', 'frustrated', 'angry', 'disappointed',
            'upset', 'sad', 'depressed', 'furious', 'irritated', 'bothered'
        }
        text_lower = text.lower()
        return sum(1 for word in negative_words if word in text_lower)
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space]
        else:
            # Simple fallback tokenization
            return re.findall(r'\b\w+\b', text.lower())
            
    def get_word_frequencies(self, texts: List[str], top_n: int = 100) -> Dict[str, int]:
        """Get word frequencies across multiple texts."""
        all_words = []
        for text in texts:
            words = self.tokenize(self.preprocess_text(text))
            all_words.extend(words)
            
        return dict(Counter(all_words).most_common(top_n))
        
    def batch_preprocess(self, texts: List[str], options: Dict[str, bool] = None) -> List[str]:
        """Preprocess multiple texts."""
        return [self.preprocess_text(text, options) for text in texts]
    
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
