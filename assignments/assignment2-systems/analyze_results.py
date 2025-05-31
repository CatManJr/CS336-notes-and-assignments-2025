#!/usr/bin/env python3
"""
Analyze and visualize distributed communication benchmark results

This script processes the JSON results from distributed_benchmark.py and
generates tables and plots for analysis.
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def load_results(results_dir="benchmark_results"):
    """Load all benchmark results from JSON files"""
    all_results = []
    
    json_files = glob.glob(f"{results_dir}/*.json")
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.extend(results)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def create_summary_table(results):
    """Create a summary table of all results"""
    if not results:
        print("No results to analyze")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Extract timing statistics
    df['mean_time_ms'] = df['timing'].apply(lambda x: x['mean_ms'])
    df['std_time_ms'] = df['timing'].apply(lambda x: x['std_ms'])
    df['min_time_ms'] = df['timing'].apply(lambda x: x['min_ms'])
    
    # Group by configuration and size
    summary = df.groupby(['backend', 'device', 'world_size', 'size_mb']).agg({
        'mean_time_ms': ['mean', 'std'],
        'bandwidth_mb_per_sec': ['mean', 'std'],
        'min_time_ms': 'mean'
    }).round(3)
    
    return summary

def plot_bandwidth_comparison(results, save_path="bandwidth_comparison.png"):
    """Plot bandwidth comparison across different configurations"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Create configuration label
    df['config'] = df['backend'] + '+' + df['device'] + f'_{df["world_size"]}proc'
    
    plt.figure(figsize=(12, 8))
    
    # Plot bandwidth vs data size for each configuration
    configs = df['config'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(configs)))
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config]
        grouped = config_data.groupby('size_mb')['bandwidth_mb_per_sec'].agg(['mean', 'std'])
        
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                    label=config, marker='o', capsize=5, color=colors[i])
    
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Bandwidth (MB/s)')
    plt.title('All-Reduce Bandwidth vs Data Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Bandwidth comparison plot saved to {save_path}")

def plot_latency_comparison(results, save_path="latency_comparison.png"):
    """Plot latency comparison across different configurations"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    df['config'] = df['backend'] + '+' + df['device'] + f'_{df["world_size"]}proc'
    df['mean_time_ms'] = df['timing'].apply(lambda x: x['mean_ms'])
    df['std_time_ms'] = df['timing'].apply(lambda x: x['std_ms'])
    
    plt.figure(figsize=(12, 8))
    
    # Plot latency vs data size
    configs = df['config'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(configs)))
    
    for i, config in enumerate(configs):
        config_data = df[df['config'] == config]
        grouped = config_data.groupby('size_mb')[['mean_time_ms', 'std_time_ms']].mean()
        
        plt.errorbar(grouped.index, grouped['mean_time_ms'], yerr=grouped['std_time_ms'],
                    label=config, marker='s', capsize=5, color=colors[i])
    
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Latency (ms)')
    plt.title('All-Reduce Latency vs Data Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Latency comparison plot saved to {save_path}")

def plot_scaling_analysis(results, save_path="scaling_analysis.png"):
    """Plot scaling analysis: how performance changes with number of processes"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    df['mean_time_ms'] = df['timing'].apply(lambda x: x['mean_ms'])
    
    # Group by backend+device and data size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    backend_device_combos = df.groupby(['backend', 'device']).size().index.tolist()
    
    for idx, (backend, device) in enumerate(backend_device_combos[:4]):  # Max 4 subplots
        if idx >= len(axes):
            break
            
        subset = df[(df['backend'] == backend) & (df['device'] == device)]
        
        # Plot for different data sizes
        sizes = sorted(subset['size_mb'].unique())
        for size in sizes:
            size_data = subset[subset['size_mb'] == size]
            grouped = size_data.groupby('world_size')['mean_time_ms'].mean()
            
            axes[idx].plot(grouped.index, grouped.values, marker='o', label=f'{size:.0f} MB')
        
        axes[idx].set_xlabel('Number of Processes')
        axes[idx].set_ylabel('Latency (ms)')
        axes[idx].set_title(f'{backend.upper()} + {device.upper()}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')
    
    # Hide unused subplots
    for idx in range(len(backend_device_combos), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Scaling Analysis: Latency vs Number of Processes')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Scaling analysis plot saved to {save_path}")

def generate_report(results, output_file="benchmark_report.md"):
    """Generate a markdown report with analysis"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    df['mean_time_ms'] = df['timing'].apply(lambda x: x['mean_ms'])
    
    with open(output_file, 'w') as f:
        f.write("# Distributed Communication Benchmark Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        
        summary_table = create_summary_table(results)
        if summary_table is not None:
            f.write("### Performance by Configuration\n\n")
            f.write(summary_table.to_string())
            f.write("\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Backend comparison
        backend_perf = df.groupby('backend')['bandwidth_mb_per_sec'].mean()
        f.write("### Backend Performance\n")
        for backend, bandwidth in backend_perf.items():
            f.write(f"- **{backend.upper()}**: {bandwidth:.1f} MB/s average bandwidth\n")
        f.write("\n")
        
        # Scaling analysis
        f.write("### Scaling Behavior\n")
        scaling_data = df.groupby(['backend', 'device', 'world_size'])['mean_time_ms'].mean()
        f.write("Average latency by process count:\n")
        for (backend, device), group in df.groupby(['backend', 'device']):
            f.write(f"\n**{backend.upper()} + {device.upper()}:**\n")
            proc_latency = group.groupby('world_size')['mean_time_ms'].mean()
            for procs, latency in proc_latency.items():
                f.write(f"- {procs} processes: {latency:.2f} ms\n")
        
        f.write("\n## Analysis Notes\n\n")
        f.write("- All-reduce operations show expected scaling behavior\n")
        f.write("- Bandwidth generally increases with data size due to better amortization of latency\n")
        f.write("- NCCL+GPU typically outperforms Gloo+CPU for large data sizes\n")
        f.write("- Communication overhead becomes more significant with more processes\n")
    
    print(f"Report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze distributed benchmark results')
    parser.add_argument('--results_dir', default='benchmark_results', 
                        help='Directory containing benchmark result JSON files')
    parser.add_argument('--output_dir', default='analysis_output',
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load results
    print("Loading benchmark results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Please run the benchmarks first.")
        return
    
    print(f"Loaded {len(results)} benchmark results")
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary = create_summary_table(results)
    if summary is not None:
        print("\nSummary Table:")
        print(summary)
        
        # Save summary table
        summary.to_csv(f"{args.output_dir}/summary_table.csv")
        print(f"Summary table saved to {args.output_dir}/summary_table.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_bandwidth_comparison(results, f"{args.output_dir}/bandwidth_comparison.png")
    plot_latency_comparison(results, f"{args.output_dir}/latency_comparison.png")
    plot_scaling_analysis(results, f"{args.output_dir}/scaling_analysis.png")
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, f"{args.output_dir}/benchmark_report.md")
    
    print(f"\nAnalysis complete! Check the '{args.output_dir}' directory for results.")

if __name__ == '__main__':
    main()