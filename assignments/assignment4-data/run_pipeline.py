#!/usr/bin/env python3
"""
Master script to run the complete data filtering and training pipeline.
This script demonstrates the full workflow from data filtering to model training simulation.
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys


def run_command(cmd: list, description: str):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print(f"SUCCESS: {description} completed in {elapsed:.1f} seconds")
    if result.stdout:
        print(f"Output: {result.stdout}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete data filtering pipeline")
    parser.add_argument("--skip_filtering", action="store_true",
                       help="Skip data filtering (use existing filtered data)")
    parser.add_argument("--skip_tokenization", action="store_true", 
                       help="Skip tokenization (use existing tokenized data)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training simulation")
    parser.add_argument("--sample_size", type=int, default=10,
                       help="Number of sample files to create")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Define paths
    sample_data_dir = Path("./sample_data")
    filtered_data_dir = Path("./filtered_data") 
    results_dir = Path("./results")
    
    combined_filtered_file = filtered_data_dir / "combined_filtered.txt"
    tokenized_file = results_dir / "tokenized_data.bin"
    
    print("="*60)
    print("DATA FILTERING AND TRAINING PIPELINE")
    print("="*60)
    print(f"Sample data directory: {sample_data_dir}")
    print(f"Filtered data directory: {filtered_data_dir}")
    print(f"Results directory: {results_dir}")
    
    # Create directories
    for directory in [sample_data_dir, filtered_data_dir, results_dir]:
        directory.mkdir(exist_ok=True)
    
    # Step 1: Data Filtering
    if not args.skip_filtering:
        print(f"\nSTEP 1: DATA FILTERING")
        
        # Check if we need to create sample data
        sample_files = list(sample_data_dir.glob("*.txt"))
        if not sample_files:
            print("No sample files found. Creating sample data...")
            cmd = [
                "python", "scripts/filter_wet_files.py",
                "--create_sample",
                "--input_dir", str(sample_data_dir),
                "--output_dir", str(filtered_data_dir),
                "--max_workers", str(args.max_workers)
            ]
        else:
            print(f"Found {len(sample_files)} sample files. Running filtering...")
            cmd = [
                "python", "scripts/filter_wet_files.py",
                "--input_dir", str(sample_data_dir),
                "--output_dir", str(filtered_data_dir),
                "--max_workers", str(args.max_workers),
                "--file_pattern", "*.txt"
            ]
        
        if not run_command(cmd, "Data Filtering"):
            print("ERROR: Data filtering failed!")
            return 1
            
        print("SUCCESS: Data filtering completed!")
        
        # Check if filtered data exists
        if not combined_filtered_file.exists():
            print(f"ERROR: Expected filtered data file {combined_filtered_file} not found!")
            return 1
            
        print(f"OUTPUT: Filtered data saved to: {combined_filtered_file}")
        
    else:
        print(f"\nSTEP 1 SKIPPED: Using existing filtered data")
        if not combined_filtered_file.exists():
            print(f"ERROR: Filtered data file {combined_filtered_file} not found!")
            return 1
    
    # Step 2: Data Inspection
    print(f"\nSTEP 2: DATA INSPECTION")
    
    cmd = [
        "python", "scripts/inspect_data.py",
        "--filtered_data", str(combined_filtered_file),
        "--stats_dir", str(filtered_data_dir),
        "--num_examples", "5"
    ]
    
    if not run_command(cmd, "Data Inspection"):
        print("WARNING: Data inspection failed, but continuing...")
    else:
        print("SUCCESS: Data inspection completed!")
    
    # Step 3: Tokenization
    if not args.skip_tokenization:
        print(f"\nSTEP 3: TOKENIZATION")
        
        cmd = [
            "python", "scripts/tokenize_data.py",
            "--input_path", str(combined_filtered_file),
            "--output_path", str(tokenized_file),
            "--tokenizer", "gpt2",
            "--max_workers", str(args.max_workers)
        ]
        
        if not run_command(cmd, "Tokenization"):
            print("ERROR: Tokenization failed!")
            return 1
            
        print("SUCCESS: Tokenization completed!")
        print(f"OUTPUT: Tokenized data saved to: {tokenized_file}")
        
    else:
        print(f"\nSTEP 3 SKIPPED: Using existing tokenized data")
        if not tokenized_file.exists():
            print(f"ERROR: Tokenized data file {tokenized_file} not found!")
            return 1
    
    # Step 4: Training Simulation
    if not args.skip_training:
        print(f"\nSTEP 4: TRAINING SIMULATION")
        
        training_output_dir = results_dir / "training_simulation"
        
        cmd = [
            "python", "scripts/simulate_training.py",
            "--dataset", str(tokenized_file),
            "--output_dir", str(training_output_dir),
            "--total_steps", "50000",  # Reduced for faster simulation
            "--eval_interval", "1000",
            "--learning_rate", "5e-4"
        ]
        
        if not run_command(cmd, "Training Simulation"):
            print("ERROR: Training simulation failed!")
            return 1
            
        print("SUCCESS: Training simulation completed!")
        print(f"OUTPUT: Training results saved to: {training_output_dir}")
        
    else:
        print(f"\nSTEP 4 SKIPPED: Training simulation disabled")
    
    # Summary
    print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    if combined_filtered_file.exists():
        file_size = combined_filtered_file.stat().st_size / (1024 * 1024)
        print(f"Filtered data: {combined_filtered_file} ({file_size:.1f} MB)")
    
    if tokenized_file.exists():
        file_size = tokenized_file.stat().st_size / (1024 * 1024)
        print(f"Tokenized data: {tokenized_file} ({file_size:.1f} MB)")
    
    training_results_file = results_dir / "training_simulation" / "training_results.json"
    if training_results_file.exists():
        import json
        with open(training_results_file) as f:
            results = json.load(f)
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Best validation perplexity: {results['best_val_perplexity']:.2f}")
        print(f"Data quality score: {results['data_quality_score']:.3f}")
    
    print("\nOutput files:")
    print(f"  - Filtered data: {combined_filtered_file}")
    print(f"  - Tokenized data: {tokenized_file}")
    print(f"  - Training results: {results_dir}/training_simulation/")
    print(f"  - Filtering stats: {filtered_data_dir}/*.stats.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())