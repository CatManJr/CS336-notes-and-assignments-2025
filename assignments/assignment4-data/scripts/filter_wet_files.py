#!/usr/bin/env python3
"""
Data filtering pipeline for Common Crawl WET files.
Since we don't have access to the Stanford cluster, this script can work with:
1. Sample WET files if available
2. Simulated text data for testing the pipeline
3. Any text files as input for demonstration
"""

import os
import gzip
import json
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import multiprocessing as mp

# Import our filtering functions
from cs336_data.extract import (
    extract_text_from_html_bytes,
    identify_language,
    mask_emails,
    mask_phone_numbers, 
    mask_ips,
    gopher_quality_filter,
    classify_quality,
    classify_nsfw,
    classify_toxic_speech
)

try:
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    FASTWARC_AVAILABLE = True
except ImportError:
    FASTWARC_AVAILABLE = False
    print("Warning: fastwarc not available. Will use text file simulation mode.")

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False
    print("Warning: tldextract not available. Domain filtering disabled.")


class FilterStats:
    """Track filtering statistics"""
    def __init__(self):
        self.total_documents = 0
        self.after_language_filter = 0
        self.after_quality_filter = 0
        self.after_gopher_filter = 0
        self.after_nsfw_filter = 0
        self.after_toxic_filter = 0
        self.after_pii_masking = 0
        self.after_domain_filter = 0
        self.final_documents = 0
        self.total_tokens = 0

    def print_stats(self):
        """Print filtering statistics"""
        print("\n" + "="*50)
        print("FILTERING STATISTICS")
        print("="*50)
        print(f"Total input documents: {self.total_documents:,}")
        print(f"After language filter (English): {self.after_language_filter:,} ({self.after_language_filter/max(1,self.total_documents)*100:.1f}%)")
        print(f"After quality filter (wiki-like): {self.after_quality_filter:,} ({self.after_quality_filter/max(1,self.total_documents)*100:.1f}%)")
        print(f"After Gopher filters: {self.after_gopher_filter:,} ({self.after_gopher_filter/max(1,self.total_documents)*100:.1f}%)")
        print(f"After NSFW filter: {self.after_nsfw_filter:,} ({self.after_nsfw_filter/max(1,self.total_documents)*100:.1f}%)")
        print(f"After toxic speech filter: {self.after_toxic_filter:,} ({self.after_toxic_filter/max(1,self.total_documents)*100:.1f}%)")
        print(f"After domain filter: {self.after_domain_filter:,} ({self.after_domain_filter/max(1,self.total_documents)*100:.1f}%)")
        print(f"Final documents after PII masking: {self.final_documents:,} ({self.final_documents/max(1,self.total_documents)*100:.1f}%)")
        print(f"Total tokens in final dataset: {self.total_tokens:,}")
        print("="*50)


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    if not TLDEXTRACT_AVAILABLE:
        # Simple fallback
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""
    
    try:
        extracted = tldextract.extract(url)
        return f"{extracted.domain}.{extracted.suffix}"
    except:
        return ""


def should_keep_domain(domain: str) -> bool:
    """
    Filter domains based on quality indicators.
    Keep domains that are likely to contain high-quality content.
    """
    if not domain:
        return False
    
    # High-quality domains (news, education, reference sites)
    good_domains = {
        'wikipedia.org', 'reddit.com', 'stackoverflow.com', 'github.com',
        'arxiv.org', 'news.ycombinator.com', 'medium.com', 'substack.com',
        'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'bbc.com',
        'reuters.com', 'bloomberg.com', 'economist.com', 'atlantic.com',
        'nature.com', 'science.org', 'ieee.org', 'acm.org'
    }
    
    # Check if it's a known good domain
    if domain.lower() in good_domains:
        return True
    
    # Check for educational domains
    if domain.endswith('.edu') or domain.endswith('.ac.uk'):
        return True
    
    # Check for government domains
    if domain.endswith('.gov'):
        return True
    
    # Reject known low-quality domains
    bad_domains = {
        'pinterest.com', 'instagram.com', 'facebook.com', 'twitter.com',
        'tiktok.com', 'snapchat.com', 'tumblr.com'
    }
    
    if domain.lower() in bad_domains:
        return False
    
    # Default: keep unknown domains but be conservative
    return True


def filter_single_document(doc_data: Tuple[str, str, str]) -> Tuple[bool, str, Dict]:
    """
    Filter a single document through all filters.
    
    Args:
        doc_data: (url, text_content, source_file)
        
    Returns:
        (keep_document, filtered_text, filter_results)
    """
    url, text, source_file = doc_data
    
    if not text or len(text.strip()) < 100:
        return False, "", {"reason": "too_short"}
    
    filter_results = {"source": source_file, "url": url}
    
    # 1. Language identification
    try:
        lang, lang_conf = identify_language(text)
        filter_results["language"] = lang
        filter_results["language_confidence"] = lang_conf
        
        if lang != "en" or lang_conf < 0.5:
            filter_results["reason"] = "non_english"
            return False, "", filter_results
    except Exception as e:
        filter_results["reason"] = f"language_error: {e}"
        return False, "", filter_results
    
    # 2. Domain filtering
    domain = extract_domain(url)
    filter_results["domain"] = domain
    
    if not should_keep_domain(domain):
        filter_results["reason"] = "bad_domain"
        return False, "", filter_results
    
    # 3. Quality classification
    try:
        quality_label, quality_score = classify_quality(text)
        filter_results["quality_label"] = quality_label
        filter_results["quality_score"] = quality_score
        
        if quality_label != "wiki" or quality_score < 0.6:
            filter_results["reason"] = "low_quality"
            return False, "", filter_results
    except Exception as e:
        filter_results["reason"] = f"quality_error: {e}"
        return False, "", filter_results
    
    # 4. Gopher quality filters
    if not gopher_quality_filter(text):
        filter_results["reason"] = "gopher_filter"
        return False, "", filter_results
    
    # 5. NSFW filtering
    try:
        nsfw_label, nsfw_score = classify_nsfw(text)
        filter_results["nsfw_label"] = nsfw_label
        filter_results["nsfw_score"] = nsfw_score
        
        if nsfw_label == "nsfw":
            filter_results["reason"] = "nsfw_content"
            return False, "", filter_results
    except Exception as e:
        filter_results["reason"] = f"nsfw_error: {e}"
        return False, "", filter_results
    
    # 6. Toxic speech filtering
    try:
        toxic_label, toxic_score = classify_toxic_speech(text)
        filter_results["toxic_label"] = toxic_label
        filter_results["toxic_score"] = toxic_score
        
        if toxic_label == "toxic":
            filter_results["reason"] = "toxic_speech"
            return False, "", filter_results
    except Exception as e:
        filter_results["reason"] = f"toxic_error: {e}"
        return False, "", filter_results
    
    # 7. PII masking
    try:
        masked_text = text
        
        # Mask emails
        masked_text, email_count = mask_emails(masked_text)
        filter_results["emails_masked"] = email_count
        
        # Mask phone numbers
        masked_text, phone_count = mask_phone_numbers(masked_text)
        filter_results["phones_masked"] = phone_count
        
        # Mask IP addresses
        masked_text, ip_count = mask_ips(masked_text)
        filter_results["ips_masked"] = ip_count
        
    except Exception as e:
        filter_results["reason"] = f"pii_error: {e}"
        return False, "", filter_results
    
    filter_results["reason"] = "kept"
    filter_results["final_length"] = len(masked_text)
    
    return True, masked_text, filter_results


def process_wet_file(input_path: str, output_dir: str) -> Tuple[str, int, List[Dict]]:
    """
    Process a single WET file and extract filtered documents.
    
    Returns:
        (output_file_path, num_documents_kept, filter_stats)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{input_path.stem}.filtered.txt"
    stats_file = output_dir / f"{input_path.stem}.stats.json"
    
    documents = []
    filter_stats = []
    
    try:
        if FASTWARC_AVAILABLE and input_path.suffix == '.gz':
            # Process actual WET file
            with gzip.open(input_path, 'rb') as f:
                for record in ArchiveIterator(f):
                    if record.record_type == WarcRecordType.conversion:
                        url = record.headers.get('WARC-Target-URI', '')
                        text = record.reader().read().decode('utf-8', errors='ignore')
                        documents.append((url, text, str(input_path)))
        else:
            # Process as text file (simulation mode)
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Split into chunks (simulating documents)
                chunks = content.split('\n\n')
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 100:
                        fake_url = f"http://example.com/doc_{i}"
                        documents.append((fake_url, chunk.strip(), str(input_path)))
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return str(output_file), 0, []
    
    # Filter documents
    kept_documents = []
    
    for doc_data in documents:
        keep, filtered_text, results = filter_single_document(doc_data)
        filter_stats.append(results)
        
        if keep:
            kept_documents.append(filtered_text)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in kept_documents:
            f.write(doc + '\n\n')
    
    # Write stats
    with open(stats_file, 'w') as f:
        json.dump(filter_stats, f, indent=2)
    
    return str(output_file), len(kept_documents), filter_stats


def create_sample_data(output_dir: str, num_files: int = 10):
    """Create sample data files for testing when real WET files aren't available"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample texts of varying quality
    sample_texts = [
        # High quality content
        """
        Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence based on the idea that systems can learn from data, 
        identify patterns and make decisions with minimal human intervention. Machine learning algorithms 
        build a mathematical model based on training data in order to make predictions or decisions 
        without being explicitly programmed to do so.
        
        The process typically involves collecting data, preparing and cleaning the data, choosing a model, 
        training the algorithm, evaluating the model, parameter tuning and prediction. There are several 
        types of machine learning including supervised learning, unsupervised learning, and reinforcement learning.
        """,
        
        # Medium quality content  
        """
        Python is a programming language that was created in the late 1980s by Guido van Rossum. 
        It has become one of the most popular programming languages in the world. Python is known 
        for its simple and readable syntax, which makes it easy for beginners to learn.
        
        Python can be used for many different types of applications including web development, 
        data analysis, artificial intelligence, and automation. Many companies use Python including 
        Google, Netflix, and Instagram.
        """,
        
        # Low quality content (should be filtered out)
        """
        Click here to sign up for our newsletter! Get the latest deals and promotions sent directly 
        to your inbox. Contact us at support@example.com or call (555) 123-4567 for more information.
        
        Copyright © 2023 Example Company. All rights reserved. 
        Powered by ExampleCMS. Privacy Policy | Terms of Service
        """,
        
        # Academic content
        """
        The transformer architecture, introduced by Vaswani et al. in 2017, has revolutionized 
        natural language processing. The key innovation was the attention mechanism, which allows 
        the model to focus on different parts of the input sequence when generating each output token.
        
        The attention mechanism computes a weighted sum of all input representations, where the weights 
        are determined by the compatibility between a query and key vectors. This allows the model to 
        capture long-range dependencies more effectively than recurrent neural networks.
        """,
    ]
    
    for i in range(num_files):
        file_path = output_dir / f"sample_wet_{i:03d}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            # Mix different quality content
            for j, text in enumerate(sample_texts):
                f.write(f"Document {i}_{j}\n")
                f.write(text.strip() + '\n\n')
    
    print(f"Created {num_files} sample files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Filter WET files for language modeling")
    parser.add_argument("--input_dir", type=str, default="./sample_data", 
                       help="Directory containing WET files or text files")
    parser.add_argument("--output_dir", type=str, default="./filtered_data",
                       help="Output directory for filtered data")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample data for testing")
    parser.add_argument("--file_pattern", type=str, default="*.txt",
                       help="Pattern to match input files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_data(input_dir)
    
    # Find input files
    if args.file_pattern.endswith('.gz'):
        input_files = list(input_dir.glob("*.warc.wet.gz"))
        if not input_files:
            input_files = list(input_dir.glob("*.gz"))
    else:
        input_files = list(input_dir.glob(args.file_pattern))
    
    if not input_files:
        print(f"No files found in {input_dir} matching pattern {args.file_pattern}")
        if not args.create_sample:
            print("Try running with --create_sample to generate test data")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    # Set up parallel processing
    max_workers = args.max_workers or min(len(input_files), mp.cpu_count())
    print(f"Using {max_workers} workers")
    
    # Process files in parallel
    futures = []
    all_stats = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        for input_file in input_files:
            future = executor.submit(process_wet_file, str(input_file), str(output_dir))
            futures.append(future)
        
        # Collect results
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(input_files), desc="Processing files"):
            try:
                output_file, num_docs, stats = future.result()
                all_stats.extend(stats)
                print(f"Processed file -> {num_docs} documents kept")
            except Exception as e:
                print(f"Error processing file: {e}")
    
    # Combine all filtered files into one
    combined_file = output_dir / "combined_filtered.txt"
    total_docs = 0
    
    with open(combined_file, 'w', encoding='utf-8') as outf:
        for output_file in output_dir.glob("*.filtered.txt"):
            with open(output_file, 'r', encoding='utf-8') as inf:
                content = inf.read()
                if content.strip():
                    outf.write(content)
                    total_docs += content.count('\n\n')
    
    # Calculate and print statistics
    stats = FilterStats()
    stats.total_documents = len(all_stats)
    
    # Count documents after each filter
    for stat in all_stats:
        if stat.get("language") == "en":
            stats.after_language_filter += 1
        if stat.get("quality_label") == "wiki":
            stats.after_quality_filter += 1
        if stat.get("reason") not in ["gopher_filter"]:
            stats.after_gopher_filter += 1
        if stat.get("nsfw_label") == "non-nsfw":
            stats.after_nsfw_filter += 1
        if stat.get("toxic_label") == "non-toxic":
            stats.after_toxic_filter += 1
        if stat.get("reason") not in ["bad_domain"]:
            stats.after_domain_filter += 1
        if stat.get("reason") == "kept":
            stats.final_documents += 1
    
    # Estimate tokens (rough approximation: 1 token per 4 characters)
    if combined_file.exists():
        with open(combined_file, 'r', encoding='utf-8') as f:
            content = f.read()
            stats.total_tokens = len(content) // 4
    
    stats.print_stats()
    
    print(f"\nFiltered data written to: {combined_file}")
    print(f"Individual file stats saved in: {output_dir}/*.stats.json")


if __name__ == "__main__":
    main()