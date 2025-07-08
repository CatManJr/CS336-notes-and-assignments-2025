#!/usr/bin/env python3
"""
Inspect filtered data to understand quality and content.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_stats_files(stats_dir: Path) -> List[Dict]:
    """Load all stats files from the filtering process"""
    all_stats = []
    for stats_file in stats_dir.glob("*.stats.json"):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            all_stats.extend(stats)
    return all_stats


def analyze_filtered_examples(filtered_file: Path, num_examples: int = 5):
    """Analyze random examples from filtered data"""
    print(f"\n{'='*60}")
    print("ANALYSIS OF FILTERED DATA (KEPT EXAMPLES)")
    print(f"{'='*60}")
    
    with open(filtered_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into documents (assuming they're separated by double newlines)
    documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
    
    if len(documents) < num_examples:
        print(f"Only {len(documents)} documents available, analyzing all of them")
        sample_docs = documents
    else:
        sample_docs = random.sample(documents, num_examples)
    
    for i, doc in enumerate(sample_docs, 1):
        print(f"\n--- KEPT EXAMPLE {i} ---")
        
        # Show first 500 characters
        preview = doc[:500]
        if len(doc) > 500:
            preview += "..."
        
        print(f"Length: {len(doc)} characters")
        print(f"Preview: {preview}")
        
        # Simple quality assessment
        quality_indicators = {
            'has_punctuation': any(c in doc for c in '.!?'),
            'has_multiple_sentences': doc.count('.') > 1,
            'reasonable_length': 100 < len(doc) < 10000,
            'mixed_case': any(c.isupper() for c in doc) and any(c.islower() for c in doc),
            'contains_numbers': any(c.isdigit() for c in doc),
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        print(f"Quality indicators: {quality_indicators}")
        print(f"Quality score: {quality_score:.2f}")
        
        # Assessment for language modeling
        if quality_score > 0.7:
            assessment = "EXCELLENT - Well-structured text suitable for language modeling"
        elif quality_score > 0.5:
            assessment = "GOOD - Reasonable quality for language modeling"
        elif quality_score > 0.3:
            assessment = "FAIR - May be useful but has quality issues"
        else:
            assessment = "POOR - Questionable value for language modeling"
        
        print(f"Assessment: {assessment}")


def analyze_discarded_examples(all_stats: List[Dict], num_examples: int = 5):
    """Analyze examples that were discarded by filters"""
    print(f"\n{'='*60}")
    print("ANALYSIS OF DISCARDED DATA (REMOVED EXAMPLES)")
    print(f"{'='*60}")
    
    # Group by rejection reason
    discarded_by_reason = {}
    for stat in all_stats:
        if stat.get('reason') != 'kept':
            reason = stat.get('reason', 'unknown')
            if reason not in discarded_by_reason:
                discarded_by_reason[reason] = []
            discarded_by_reason[reason].append(stat)
    
    print(f"\nRejection reasons and counts:")
    for reason, examples in discarded_by_reason.items():
        print(f"  {reason}: {len(examples)} examples")
    
    # Sample examples from different rejection reasons
    analyzed_count = 0
    for reason, examples in discarded_by_reason.items():
        if analyzed_count >= num_examples:
            break
        
        sample_example = random.choice(examples)
        analyzed_count += 1
        
        print(f"\n--- DISCARDED EXAMPLE {analyzed_count} ---")
        print(f"Rejection reason: {reason}")
        print(f"Source: {sample_example.get('source', 'unknown')}")
        print(f"URL: {sample_example.get('url', 'unknown')}")
        print(f"Domain: {sample_example.get('domain', 'unknown')}")
        
        if 'language' in sample_example:
            print(f"Language: {sample_example['language']} (confidence: {sample_example.get('language_confidence', 'N/A')})")
        
        if 'quality_label' in sample_example:
            print(f"Quality: {sample_example['quality_label']} (score: {sample_example.get('quality_score', 'N/A')})")
        
        # Assessment of rejection
        justified_reasons = {
            'non_english': "JUSTIFIED - Non-English content not suitable for English LM",
            'low_quality': "JUSTIFIED - Low quality content would hurt model performance",
            'gopher_filter': "JUSTIFIED - Failed basic readability criteria",
            'nsfw_content': "JUSTIFIED - Inappropriate content should be filtered",
            'toxic_speech': "JUSTIFIED - Toxic content would teach harmful patterns",
            'bad_domain': "JUSTIFIED - Low-quality domain likely to contain poor content",
            'too_short': "JUSTIFIED - Too short to be useful for language modeling"
        }
        
        assessment = justified_reasons.get(reason, "QUESTIONABLE - May have been useful content")
        print(f"Rejection assessment: {assessment}")


def print_overall_statistics(all_stats: List[Dict]):
    """Print overall filtering statistics"""
    print(f"\n{'='*60}")
    print("OVERALL FILTERING STATISTICS")
    print(f"{'='*60}")
    
    total_docs = len(all_stats)
    if total_docs == 0:
        print("No statistics available")
        return
    
    # Count by reason
    reason_counts = {}
    for stat in all_stats:
        reason = stat.get('reason', 'unknown')
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    print(f"Total documents processed: {total_docs:,}")
    print(f"Documents kept: {reason_counts.get('kept', 0):,} ({reason_counts.get('kept', 0)/total_docs*100:.1f}%)")
    print(f"Documents discarded: {total_docs - reason_counts.get('kept', 0):,} ({(total_docs - reason_counts.get('kept', 0))/total_docs*100:.1f}%)")
    
    print("\nRejection breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        if reason != 'kept':
            print(f"  {reason}: {count:,} ({count/total_docs*100:.1f}%)")
    
    # Filter efficiency analysis
    print("\nFilter efficiency analysis:")
    filter_order = [
        ('too_short', 'Length filter'),
        ('language_error', 'Language detection error'),
        ('non_english', 'Language filter (non-English)'),
        ('bad_domain', 'Domain filter'),
        ('low_quality', 'Quality classifier'),
        ('gopher_filter', 'Gopher quality rules'),
        ('nsfw_content', 'NSFW filter'),
        ('toxic_speech', 'Toxic speech filter'),
    ]
    
    remaining_docs = total_docs
    for reason_key, filter_name in filter_order:
        removed = reason_counts.get(reason_key, 0)
        if removed > 0:
            print(f"  {filter_name}: removed {removed:,} ({removed/total_docs*100:.1f}% of total)")
            remaining_docs -= removed
    
    print(f"  Final kept: {reason_counts.get('kept', 0):,} ({reason_counts.get('kept', 0)/total_docs*100:.1f}% of total)")


def main():
    parser = argparse.ArgumentParser(description="Inspect filtered data quality")
    parser.add_argument("--filtered_data", type=str, required=True,
                       help="Path to filtered data file")
    parser.add_argument("--stats_dir", type=str, required=True,
                       help="Directory containing stats files")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of examples to analyze")
    
    args = parser.parse_args()
    
    filtered_file = Path(args.filtered_data)
    stats_dir = Path(args.stats_dir)
    
    if not filtered_file.exists():
        print(f"Filtered data file {filtered_file} does not exist")
        return
    
    if not stats_dir.exists():
        print(f"Stats directory {stats_dir} does not exist")
        return
    
    # Load all statistics
    print("Loading filtering statistics...")
    all_stats = load_stats_files(stats_dir)
    
    if not all_stats:
        print("No statistics files found")
        return
    
    # Print overall statistics
    print_overall_statistics(all_stats)
    
    # Analyze kept examples
    analyze_filtered_examples(filtered_file, args.num_examples)
    
    # Analyze discarded examples
    analyze_discarded_examples(all_stats, args.num_examples)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()