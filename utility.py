

# Common functions for cleaning the text data
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Global objects (avoid re-creating inside loops)
STOPWORDS = set(stopwords.words('english'))
STEMMER = LancasterStemmer()
LEMMATIZER = WordNetLemmatizer()

# --- Text cleaning functions ---

def clean_special_characters(text, remove_digits=True):
    """Remove special characters from text"""
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def lowercase_tokens(tokens):
    """Convert list of tokens to lowercase"""
    return [token.lower() for token in tokens]

def remove_punct_and_special_chars(tokens):
    """Remove punctuation and special characters from list of tokens"""
    return [
        clean_special_characters(re.sub(r'[^\w\s]', '', token))
        for token in tokens
        if clean_special_characters(re.sub(r'[^\w\s]', '', token)) != ''
    ]

def remove_stopwords(tokens):
    """Remove stopwords from list of tokens"""
    return [token for token in tokens if token not in STOPWORDS]

def stem_tokens(tokens):
    """Stem tokens using Lancaster Stemmer"""
    return [STEMMER.stem(token) for token in tokens]

def lemmatize_tokens(tokens):
    """Lemmatize tokens using WordNetLemmatizer (verb pos)"""
    return [LEMMATIZER.lemmatize(token, pos='v') for token in tokens]

def normalize_tokens(tokens):
    """Pipeline: lowercase -> punctuation removal -> stopword removal"""
    tokens = lowercase_tokens(tokens)
    tokens = remove_punct_and_special_chars(tokens)
    tokens = remove_stopwords(tokens)
    return tokens

def lemmatize_normalized(tokens):
    """Pipeline: lemmatize normalized tokens"""
    return lemmatize_tokens(tokens)

def clean_text(text):
    """ cleaning pipeline: raw text -> clean string """
    text = clean_special_characters(text)
    tokens = word_tokenize(text)

    tokens = normalize_tokens(tokens)
    tokens = lemmatize_normalized(tokens)

    return ' '.join(tokens)