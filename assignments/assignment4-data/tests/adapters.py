from __future__ import annotations

import os
from typing import Any

from cs336_data.extract import (
    extract_text_from_html_bytes,
    identify_language,
    mask_emails,
    mask_phone_numbers,
    mask_ips,

    gopher_quality_filter,
    classify_quality,
    classify_nsfw,
    classify_toxic_speech,
    exact_line_deduplication,
    minhash_deduplication,
)


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    Extract plain text from HTML bytes.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        html_bytes: Raw HTML content as bytes

    Returns:
        Extracted plain text as string, or None if extraction fails
    """
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    """
    Identify the main language in a text string.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Unicode string to analyze

    Returns:
        Tuple of (language_code, confidence_score)
    """
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    """
    Mask email addresses in text.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text that may contain email addresses

    Returns:
        Tuple of (masked_text, count)
    """
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """
    Mask phone numbers in text.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text that may contain phone numbers

    Returns:
        Tuple of (masked_text, count)
    """
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    """
    Mask IP addresses in text.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text that may contain IP addresses

    Returns:
        Tuple of (masked_text, count)
    """
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    """
    Classify text for NSFW (Not Safe For Work) content.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (label, confidence_score)
    """
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    """
    Classify text for toxic speech.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (label, confidence_score)
    """
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    """
    Classify text quality.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (label, confidence_score)
    """
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    """
    Apply Gopher quality filters to text.

    This is an adapter function that calls the main implementation
    in the cs336_data module.

    Args:
        text: Input text to evaluate

    Returns:
        True if text passes all filters, False otherwise
    """
    return gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """
    Perform exact line deduplication on input files.
    
    This is an adapter function that calls the main implementation
    in the cs336_data module.
    
    Args:
        input_files: List of input file paths
        output_directory: Output directory path
    """
    exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    Perform fuzzy document deduplication using MinHash + LSH.
    
    This is an adapter function that calls the main implementation
    in the cs336_data module.
    
    Args:
        input_files: List of input file paths
        num_hashes: Number of hash functions for minhash signatures
        num_bands: Number of bands for LSH
        ngrams: N-gram length in words
        jaccard_threshold: Jaccard similarity threshold for duplicates
        output_directory: Output directory path
    """
    minhash_deduplication(
        input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory
    )
