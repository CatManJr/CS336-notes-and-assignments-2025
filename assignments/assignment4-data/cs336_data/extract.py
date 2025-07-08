"""
Text extraction utilities for HTML content.
"""
from __future__ import annotations

import fasttext
import os
import re
import hashlib
import random
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
from typing import Set, List, Dict, Tuple
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

# Try to download NLTK data, but continue if it fails
NLTK_AVAILABLE = False
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    # NLTK not available
    NLTK_AVAILABLE = False


def get_nltk_tokenizer():
    """
    Lazy initialization of NLTK tokenizer.
    
    Returns:
        True if NLTK tokenizer is available, False otherwise
    """
    global NLTK_AVAILABLE
    if not NLTK_AVAILABLE:
        return False
    
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            return True
        except (LookupError, Exception):
            try:
                nltk.download('punkt', quiet=True)
                return True
            except Exception:
                # NLTK download failed, we'll use fallback tokenization
                return False
    except ImportError:
        return False


def simple_word_tokenize(text: str) -> list[str]:
    """
    Simple fallback word tokenization when NLTK is not available.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of word tokens
    """
    # Simple regex-based tokenization
    # Split on whitespace and punctuation, keep alphanumeric sequences
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    Extract plain text from HTML bytes.
    
    First tries to decode as UTF-8, falls back to detected encoding if that fails.
    Uses resiliparse to extract plain text from the HTML.
    
    Args:
        html_bytes: Raw HTML content as bytes
        
    Returns:
        Extracted plain text as string, or None if extraction fails
    """
    try:
        # First try UTF-8 decoding
        try:
            html_string = html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 fails, detect the encoding
            detected_encoding = detect_encoding(html_bytes)
            if detected_encoding:
                html_string = html_bytes.decode(detected_encoding, errors="replace")
            else:
                # Fallback to latin-1 if detection fails
                html_string = html_bytes.decode("latin-1", errors="replace")
        
        # Extract plain text using resiliparse
        extracted_text = extract_plain_text(html_string)
        return extracted_text
        
    except Exception:
        return None


def identify_language(text: str) -> tuple[str, float]:
    """
    Identify the main language in a text string using fastText.
    
    Args:
        text: Unicode string to analyze
        
    Returns:
        Tuple of (language_code, confidence_score) where:
        - language_code is a string like "en", "zh", etc.
        - confidence_score is a float between 0 and 1
    """
    # Try to load the fastText model from common locations
    model_paths = [
        "/data/classifiers/lid.176.bin",  # Together cluster path
        "lid.176.bin",  # Local path
        os.path.expanduser("~/lid.176.bin"),  # Home directory
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = fasttext.load_model(path)
                break
            except Exception:
                continue
    
    if model is None:
        # Download the model if not found locally
        try:
            import urllib.request
            model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            local_path = "lid.176.bin"
            if not os.path.exists(local_path):
                urllib.request.urlretrieve(model_url, local_path)
            model = fasttext.load_model(local_path)
        except Exception:
            raise RuntimeError("Could not load or download fastText language identification model")
    
    # Clean the text - remove newlines and extra whitespace
    cleaned_text = " ".join(text.split())
    
    # Predict language
    predictions = model.predict(cleaned_text, k=1)
    language_labels, confidence_scores = predictions
    
    # Extract language code from fastText label format (__label__en -> en)
    language_code = language_labels[0].replace("__label__", "")
    confidence_score = float(confidence_scores[0])
    
    # Map some common language codes for compatibility
    language_mapping = {
        "zh-cn": "zh",  # Chinese Simplified
        "zh-tw": "zh",  # Chinese Traditional
    }
    
    language_code = language_mapping.get(language_code, language_code)
    
    return language_code, confidence_score


def mask_emails(text: str) -> tuple[str, int]:
    """
    Mask email addresses in text with |||EMAIL_ADDRESS|||.
    
    Args:
        text: Input text that may contain email addresses
        
    Returns:
        Tuple of (masked_text, count) where:
        - masked_text is the text with emails replaced
        - count is the number of emails that were masked
    """
    # Email regex pattern - matches most common email formats
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Find all email matches
    matches = re.findall(email_pattern, text)
    
    # Replace emails with mask
    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    
    return masked_text, len(matches)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Mask phone numbers in text with |||PHONE_NUMBER|||.
    
    Captures common US phone number formats:
    - 1234567890
    - 123-456-7890
    - (123) 456-7890
    - (123)-456-7890
    - 123.456.7890
    
    Args:
        text: Input text that may contain phone numbers
        
    Returns:
        Tuple of (masked_text, count) where:
        - masked_text is the text with phone numbers replaced
        - count is the number of phone numbers that were masked
    """
    # Phone number patterns for common US formats
    phone_patterns = [
        r'\b\d{10}\b',  # 1234567890
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\(\d{3}\)\s*-?\s*\d{3}[\s-]\d{4}\b',  # (123) 456-7890 or (123)-456-7890
        r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
    ]
    
    masked_text = text
    total_count = 0
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, masked_text)
        total_count += len(matches)
        masked_text = re.sub(pattern, '|||PHONE_NUMBER|||', masked_text)
    
    return masked_text, total_count


def mask_ips(text: str) -> tuple[str, int]:
    """
    Mask IPv4 addresses in text with |||IP_ADDRESS|||.
    
    Args:
        text: Input text that may contain IP addresses
        
    Returns:
        Tuple of (masked_text, count) where:
        - masked_text is the text with IP addresses replaced
        - count is the number of IP addresses that were masked
    """
    # IPv4 pattern - 4 numbers (0-255) separated by dots
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    # Find all IP matches
    matches = re.findall(ip_pattern, text)
    
    # Replace IPs with mask
    masked_text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)
    
    return masked_text, len(matches)


def gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher quality filters to determine if text meets quality criteria.
    
    Filters based on:
    - Word count (50-100,000 words)
    - Mean word length (3-10 characters)
    - Lines ending with ellipsis (<30%)
    - Words with alphabetic characters (>=80%)
    
    Args:
        text: Input text to evaluate
        
    Returns:
        True if text passes all filters, False otherwise
    """
    # Tokenize text into words using NLTK
    try:
        if NLTK_AVAILABLE:
            import nltk
            words = nltk.word_tokenize(text)
        else:
            # Fallback to simple tokenization
            words = simple_word_tokenize(text)
    except:
        # Fallback to simple splitting if tokenization fails
        words = text.split()
    
    # Filter 1: Word count (50-100,000 words)
    # Count words with at least one alphabetic character
    alphabetic_words = [word for word in words if any(c.isalpha() for c in word)]
    
    if len(alphabetic_words) < 50 or len(alphabetic_words) > 100000:
        return False
    
    # Filter 2: Mean word length (3-10 characters)
    if alphabetic_words:
        mean_word_length = sum(len(word) for word in alphabetic_words) / len(alphabetic_words)
        if mean_word_length < 3 or mean_word_length > 10:
            return False
    
    # Filter 3: Lines ending with ellipsis (<30%)
    lines = text.split('\n')
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith('...'))
        ellipsis_ratio = ellipsis_lines / len(lines)
        if ellipsis_ratio > 0.3:
            return False
    
    # Filter 4: Words with alphabetic characters (>=80%)
    if words:
        alphabetic_ratio = len(alphabetic_words) / len(words)
        if alphabetic_ratio < 0.8:
            return False
    
    return True


def classify_quality(text: str) -> tuple[str, float]:
    """
    Classify text quality using a simple heuristic-based approach.
    
    This is a placeholder implementation that uses basic heuristics
    to distinguish between high-quality (wiki-like) and low-quality (cc-like) content.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (label, confidence_score) where:
        - label is "wiki" for high quality or "cc" for low quality
        - confidence_score is a float between 0 and 1
    """
    # Simple heuristics for quality classification
    score = 0.5  # Base score
    
    # Check for high-quality indicators first
    high_quality_indicators = [
        'published', 'revision', 'bibliography', 'references', 'academic',
        'philosophy', 'theory', 'analysis', 'research', 'scholarly',
        'university', 'journal', 'article', 'encyclopedia', 'definition',
        'substantive', 'intellectual', 'conceptual', 'historical'
    ]
    
    text_lower = text.lower()
    high_quality_count = sum(1 for indicator in high_quality_indicators if indicator in text_lower)
    if high_quality_count > 2:  # Multiple academic indicators
        score += 0.3
    
    # Check for structured academic content
    if any(pattern in text_lower for pattern in ['first published', 'substantive revision', 'bibliography', 'references']):
        score += 0.2
    
    # Check for numbered sections (academic structure)
    if re.search(r'\d+\.\d+\s+[A-Z]', text):  # Pattern like "1.1 Title"
        score += 0.2
    
    # Check if text passes Gopher filters (good indicator of quality)
    if gopher_quality_filter(text):
        score += 0.1
    else:
        score -= 0.2  # Penalty for failing basic quality
    
    # Check text length (good articles tend to have reasonable length)
    words = text.split()
    if 500 <= len(words) <= 10000:  # Academic articles are usually longer
        score += 0.2
    elif 100 <= len(words) <= 500:
        score += 0.1
    elif len(words) < 50:
        score -= 0.3  # Very short content is likely low quality
    
    # Check for structure indicators (paragraphs, proper sentences)
    sentences = text.split('.')
    if len(sentences) > 10:  # Substantial content
        score += 0.2
    elif len(sentences) > 5:
        score += 0.1
    
    # Check for common low-quality indicators (more comprehensive)
    low_quality_indicators = [
        'click here', 'sign up', 'log in', 'contact us', 'error 404',
        'page not found', 'javascript required', 'cookies', 'privacy policy',
        'terms of service', 'subscribe', 'newsletter', 'advertisement',
        'forum index', 'memberlist', 'usergroups', 'register', 'profile',
        'log in to check', 'powered by', 'copyright ©', 'all rights reserved',
        'faq', 'search', 'discussion forums', 'phpbb', 'meeting place'
    ]
    
    indicator_count = sum(1 for indicator in low_quality_indicators if indicator in text_lower)
    if indicator_count > 0:
        score -= 0.2 * indicator_count  # Strong penalty for each low-quality indicator
    
    # Check for navigation/UI elements (strong indicator of low quality)
    nav_patterns = [
        r'\bfaq\b.*\bsearch\b.*\bmemberlist\b',  # FAQ Search Memberlist pattern
        r'log in.*check.*private messages',
        r'powered by.*phpbb',
        r'copyright.*all rights reserved'
    ]
    
    for pattern in nav_patterns:
        if re.search(pattern, text_lower):
            score -= 0.4  # Heavy penalty for navigation patterns
    
    # Check for forum/website structural content
    if any(term in text_lower for term in ['forum', 'discussion forums', 'index', 'memberlist']):
        score -= 0.3
    
    # Check for repetitive content
    lines = text.split('\n')
    if len(lines) > 3:
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < len(lines) * 0.5:  # More than 50% duplicate lines
            score -= 0.3
    
    # Ensure score is between 0.1 and 1 (never exactly 0)
    score = max(0.1, min(1.0, score))
    
    # Classify based on score threshold
    if score >= 0.6:
        return "wiki", score
    else:
        return "cc", score


def classify_nsfw(text: str) -> tuple[str, float]:
    """
    Classify text as NSFW (Not Safe For Work) or non-nsfw.
    
    This is a simple heuristic-based implementation since we don't have access
    to the Dolma fastText models on macOS without Stanford network access.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (label, confidence_score) where:
        - label is "nsfw" or "non-nsfw"
        - confidence_score is a float between 0 and 1
    """
    # NSFW keywords and patterns (including censored versions)
    nsfw_keywords = [
        'fuck', 'shit', 'bitch', 'damn', 'ass', 'cock', 'cunt', 'whore',
        'porn', 'sex', 'nude', 'naked', 'boobs', 'penis', 'vagina',
        'masturbat', 'orgasm', 'erotic', 'xxx', 'adult', 'explicit'
    ]
    
    # Censored profanity patterns
    censored_patterns = [
        r'f\*+ck', r'sh\*+t', r'b\*+tch', r'c\*+ck', r'c\*+nt',
        r'\*ssh\*+le', r'd\*+mn', r'wh\*+re', r'b\*+st\*+rd'
    ]
    
    text_lower = text.lower()
    
    # Count NSFW keywords
    nsfw_count = 0
    for keyword in nsfw_keywords:
        nsfw_count += text_lower.count(keyword)
    
    # Count censored profanity
    censored_count = 0
    for pattern in censored_patterns:
        matches = re.findall(pattern, text_lower)
        censored_count += len(matches)
    
    # Calculate score based on frequency
    words = text.split()
    if len(words) == 0:
        return "non-nsfw", 0.5
    
    # Consider both direct and censored profanity
    total_nsfw_count = nsfw_count + censored_count
    nsfw_ratio = total_nsfw_count / len(words)
    
    # If we have any censored profanity, it's likely NSFW
    if censored_count > 0:
        confidence = min(0.9, 0.7 + censored_count * 0.1)
        return "nsfw", confidence
    
    # Higher ratio = more likely NSFW
    if nsfw_ratio > 0.05:  # More than 5% NSFW words
        confidence = min(0.9, 0.5 + nsfw_ratio * 10)
        return "nsfw", confidence
    elif nsfw_ratio > 0.01:  # 1-5% NSFW words
        confidence = 0.5 + nsfw_ratio * 5
        return "nsfw", confidence
    else:
        confidence = max(0.6, 1.0 - nsfw_ratio * 20)
        return "non-nsfw", confidence


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """
    Classify text as toxic speech or non-toxic.
    
    This is a simple heuristic-based implementation since we don't have access
    to the Dolma fastText models on macOS without Stanford network access.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (label, confidence_score) where:
        - label is "toxic" or "non-toxic" 
        - confidence_score is a float between 0 and 1
    """
    # Toxic speech indicators
    toxic_keywords = [
        'idiot', 'moron', 'stupid', 'dumb', 'retard', 'loser', 'pathetic',
        'hate', 'kill', 'die', 'murder', 'violence', 'threat',
        'fuck', 'shit', 'damn', 'asshole', 'bitch', 'bastard',
        'racist', 'nazi', 'terrorist', 'scum', 'trash', 'garbage'
    ]
    
    # Aggressive patterns
    aggressive_patterns = [
        r'\byou\s+are\s+\w*\s*(stupid|idiot|moron)',
        r'\bgo\s+(die|kill\s+yourself)',
        r'\bi\s+hate\s+you',
        r'\bshut\s+up',
        r'\bfuck\s+you',
    ]
    
    text_lower = text.lower()
    
    # Count toxic keywords
    toxic_count = 0
    for keyword in toxic_keywords:
        toxic_count += text_lower.count(keyword)
    
    # Check for aggressive patterns
    pattern_matches = 0
    for pattern in aggressive_patterns:
        if re.search(pattern, text_lower):
            pattern_matches += 1
    
    # Calculate toxicity score
    words = text.split()
    if len(words) == 0:
        return "non-toxic", 0.5
    
    toxic_ratio = toxic_count / len(words)
    
    # Combine keyword ratio and pattern matches
    toxicity_score = toxic_ratio + (pattern_matches * 0.1)
    
    if toxicity_score > 0.05 or pattern_matches > 0:
        confidence = min(0.9, 0.6 + toxicity_score * 5)
        return "toxic", confidence
    else:
        confidence = max(0.6, 1.0 - toxicity_score * 10)
        return "non-toxic", confidence


def normalize_text(text: str) -> str:
    """
    Normalize text for deduplication by:
    - Converting to lowercase
    - Removing punctuation 
    - Normalizing whitespace
    - Removing accents
    - Applying NFD unicode normalization
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # NFD unicode normalization
    text = unicodedata.normalize('NFD', text)
    
    # Remove accents
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_ngrams(text: str, n: int) -> Set[str]:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text
        n: Length of n-grams in words
        
    Returns:
        Set of n-grams
    """
    words = text.split()
    if len(words) < n:
        return set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.add(ngram)
    
    return ngrams


def hash_function(text: str, seed: int = 0) -> int:
    """
    Hash function for minhashing.
    
    Args:
        text: Text to hash
        seed: Seed for hash function
        
    Returns:
        Hash value as integer
    """
    # Use hashlib with seed for deterministic hashing
    hasher = hashlib.md5()
    hasher.update(f"{seed}:{text}".encode('utf-8'))
    return int(hasher.hexdigest(), 16)


def compute_minhash_signature(ngrams: Set[str], num_hashes: int) -> List[int]:
    """
    Compute minhash signature for a set of n-grams.
    
    Args:
        ngrams: Set of n-grams
        num_hashes: Number of hash functions to use
        
    Returns:
        Minhash signature as list of integers
    """
    if not ngrams:
        return [0] * num_hashes
    
    signature = []
    for i in range(num_hashes):
        min_hash = min(hash_function(ngram, seed=i) for ngram in ngrams)
        signature.append(min_hash)
    
    return signature


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity (0 to 1)
    """
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def exact_line_deduplication(input_files: List[os.PathLike], output_directory: os.PathLike) -> None:
    """
    Perform exact line deduplication on input files.
    
    First pass: Count frequency of each line (using hash for memory efficiency)
    Second pass: Rewrite files keeping only unique lines
    
    Args:
        input_files: List of input file paths
        output_directory: Output directory path
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First pass: count line frequencies using hashes
    line_counts = Counter()
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n\r')
                # Use hash of line as key for memory efficiency
                line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
                line_counts[line_hash] += 1
    
    # Second pass: rewrite files keeping only unique lines
    for file_path in input_files:
        file_path = Path(file_path)
        output_path = output_dir / file_path.name
        
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                original_line = line.rstrip('\n\r')
                line_hash = hashlib.md5(original_line.encode('utf-8')).hexdigest()
                
                # Only keep lines that appear exactly once in the corpus
                if line_counts[line_hash] == 1:
                    outfile.write(line)


def minhash_deduplication(
    input_files: List[os.PathLike], 
    num_hashes: int,
    num_bands: int, 
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike
) -> None:
    """
    Perform fuzzy document deduplication using MinHash + LSH.
    
    Args:
        input_files: List of input file paths
        num_hashes: Number of hash functions for minhash signatures
        num_bands: Number of bands for LSH
        ngrams: N-gram length in words
        jaccard_threshold: Jaccard similarity threshold for duplicates
        output_directory: Output directory path
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all documents and compute signatures
    documents = {}
    signatures = {}
    
    for file_path in input_files:
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Normalize text and extract n-grams
        normalized_content = normalize_text(content)
        document_ngrams = get_ngrams(normalized_content, ngrams)
        
        # Compute minhash signature
        signature = compute_minhash_signature(document_ngrams, num_hashes)
        
        documents[file_path.name] = {
            'path': file_path,
            'content': content,
            'ngrams': document_ngrams
        }
        signatures[file_path.name] = signature
    
    # LSH: Group documents into bands and find candidate pairs
    rows_per_band = num_hashes // num_bands
    candidate_pairs = set()
    
    for band_idx in range(num_bands):
        band_buckets = defaultdict(list)
        
        for doc_name, signature in signatures.items():
            # Extract band signature
            start_idx = band_idx * rows_per_band
            end_idx = start_idx + rows_per_band
            band_signature = tuple(signature[start_idx:end_idx])
            
            # Hash band signature for bucketing
            band_hash = hash(band_signature)
            band_buckets[band_hash].append(doc_name)
        
        # Add candidate pairs from same buckets
        for bucket_docs in band_buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        candidate_pairs.add((bucket_docs[i], bucket_docs[j]))
    
    # Compute true Jaccard similarities for candidate pairs
    duplicate_pairs = set()
    for doc1, doc2 in candidate_pairs:
        ngrams1 = documents[doc1]['ngrams']
        ngrams2 = documents[doc2]['ngrams']
        
        jaccard_sim = jaccard_similarity(ngrams1, ngrams2)
        if jaccard_sim >= jaccard_threshold:
            duplicate_pairs.add((doc1, doc2))
    
    # Cluster duplicates using union-find
    clusters = {}
    
    def find_root(doc):
        if doc not in clusters:
            clusters[doc] = doc
        if clusters[doc] != doc:
            clusters[doc] = find_root(clusters[doc])
        return clusters[doc]
    
    def union(doc1, doc2):
        root1 = find_root(doc1)
        root2 = find_root(doc2)
        if root1 != root2:
            clusters[root2] = root1
    
    # Build clusters from duplicate pairs
    for doc1, doc2 in duplicate_pairs:
        union(doc1, doc2)
    
    # Group documents by cluster
    cluster_groups = defaultdict(list)
    for doc_name in documents.keys():
        root = find_root(doc_name)
        cluster_groups[root].append(doc_name)
    
    # Keep one random document from each cluster
    kept_documents = set()
    for cluster_docs in cluster_groups.values():
        # Randomly select one document to keep from each cluster
        kept_doc = random.choice(cluster_docs)
        kept_documents.add(kept_doc)
    
    # Write kept documents to output directory
    for doc_name in kept_documents:
        doc_info = documents[doc_name]
        output_path = output_dir / doc_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_info['content'])