#!/usr/bin/env python3
"""
Script to analyze Expert Iteration results and generate plots.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_experiment_results(results_dir: str) -> Dict[str, Any]:
    """Load results from an Expert Iteration experiment."""
    results = {
        'config': {},
        'steps': [],
        'final_results': None
    }
    
    # Load step results
    for step in range(1, 6):  # 5 EI steps
        step_file = os.path.join(results_dir, f'ei_step_{step}', 'step_results.json')
        if os.path.exists(step_file):
            with open(step_file, 'r') as f:
                step_data = json.load(f)
                results['steps'].append(step_data)
    
    # Load final results if available
    final_file = os.path.join(results_dir, 'final_results.json')
    if os.path.exists(final_file):
        with open(final_file, 'r') as f:
            results['final_results'] = json.load(f)
    
    return results


def analyze_all_experiments(base_dir: str) -> Dict[str, Any]:
    """Analyze all Expert Iteration experiments."""
    experiments = {}
    
    if not os.path.exists(base_dir):
        print(f"Results directory {base_dir} not found")
        return experiments
    
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        if os.path.isdir(exp_path):
            print(f"Loading experiment: {exp_dir}")
            experiments[exp_dir] = load_experiment_results(exp_path)
    
    return experiments


def plot_validation_accuracy_curves(experiments: Dict[str, Any], output_dir: str):
    """Plot validation accuracy curves for all experiments."""
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_data in experiments.items():
        if exp_data['steps']:
            steps = [step['ei_step'] for step in exp_data['steps']]
            accuracies = [step['eval_results']['accuracy'] for step in exp_data['steps']]
            
            plt.plot(steps, accuracies, marker='o', label=exp_name, linewidth=2)
    
    plt.xlabel('Expert Iteration Step')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Across Expert Iteration Steps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'validation_accuracy_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved validation accuracy curves to {output_path}")


def plot_entropy_curves(experiments: Dict[str, Any], output_dir: str):
    """Plot entropy curves for all experiments."""
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_data in experiments.items():
        if exp_data['steps']:
            steps = [step['ei_step'] for step in exp_data['steps']]
            entropies = [step['entropy_stats']['mean_entropy'] for step in exp_data['steps']]
            
            plt.plot(steps, entropies, marker='s', label=exp_name, linewidth=2)
    
    plt.xlabel('Expert Iteration Step')
    plt.ylabel('Mean Response Entropy')
    plt.title('Model Response Entropy Across Expert Iteration Steps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'entropy_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved entropy curves to {output_path}")


def plot_generation_accuracy_curves(experiments: Dict[str, Any], output_dir: str):
    """Plot generation accuracy curves for all experiments."""
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_data in experiments.items():
        if exp_data['steps']:
            steps = [step['ei_step'] for step in exp_data['steps']]
            gen_accuracies = [step['eval_stats']['total_accuracy'] for step in exp_data['steps']]
            
            plt.plot(steps, gen_accuracies, marker='^', label=exp_name, linewidth=2)
    
    plt.xlabel('Expert Iteration Step')
    plt.ylabel('Generation Accuracy')
    plt.title('Generation Accuracy Across Expert Iteration Steps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'generation_accuracy_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved generation accuracy curves to {output_path}")


def generate_summary_table(experiments: Dict[str, Any], output_dir: str):
    """Generate a summary table of all experiments."""
    summary_data = []
    
    for exp_name, exp_data in experiments.items():
        if exp_data['steps']:
            # Parse configuration from experiment name
            parts = exp_name.split('_')
            rollouts = int(parts[0].replace('rollouts', ''))
            epochs = int(parts[1].replace('epochs', ''))
            batch_size = int(parts[2].replace('batch', ''))
            
            # Get final step results
            final_step = exp_data['steps'][-1]
            
            summary_data.append({
                'Experiment': exp_name,
                'Rollouts': rollouts,
                'Epochs': epochs,
                'Batch Size': batch_size,
                'Final Validation Accuracy': final_step['eval_results']['accuracy'],
                'Final Format Accuracy': final_step['eval_results']['format_accuracy'],
                'Final Generation Accuracy': final_step['eval_stats']['total_accuracy'],
                'Final Entropy': final_step['entropy_stats']['mean_entropy'],
                'Correct Responses (Final)': final_step['eval_stats']['correct_responses'],
                'Total Responses (Final)': final_step['eval_stats']['total_responses']
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Sort by validation accuracy
        df = df.sort_values('Final Validation Accuracy', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'experiment_summary.csv')
        df.to_csv(csv_path, index=False)
        
        # Create formatted table
        print("\nExperiment Summary:")
        print("=" * 120)
        print(df.to_string(index=False, float_format='%.3f'))
        print(f"\nSummary table saved to {csv_path}")
        
        return df
    
    return None


def compare_to_baseline(experiments: Dict[str, Any], baseline_accuracy: float = 0.0):
    """Compare Expert Iteration results to baseline."""
    print(f"\nComparison to Baseline (Accuracy: {baseline_accuracy:.3f}):")
    print("=" * 60)
    
    improvements = []
    for exp_name, exp_data in experiments.items():
        if exp_data['steps']:
            final_acc = exp_data['steps'][-1]['eval_results']['accuracy']
            improvement = final_acc - baseline_accuracy
            improvements.append((exp_name, final_acc, improvement))
    
    # Sort by improvement
    improvements.sort(key=lambda x: x[2], reverse=True)
    
    for exp_name, final_acc, improvement in improvements:
        status = "✓" if improvement > 0 else "✗"
        print(f"{status} {exp_name}: {final_acc:.3f} ({improvement:+.3f})")
    
    return improvements


def analyze_entropy_trends(experiments: Dict[str, Any]):
    """Analyze entropy trends across experiments."""
    print("\nEntropy Analysis:")
    print("=" * 50)
    
    for exp_name, exp_data in experiments.items():
        if exp_data['steps']:
            entropies = [step['entropy_stats']['mean_entropy'] for step in exp_data['steps']]
            
            initial_entropy = entropies[0]
            final_entropy = entropies[-1]
            entropy_change = final_entropy - initial_entropy
            
            trend = "↗" if entropy_change > 0.1 else "↘" if entropy_change < -0.1 else "→"
            
            print(f"{exp_name}:")
            print(f"  Initial entropy: {initial_entropy:.3f}")
            print(f"  Final entropy: {final_entropy:.3f}")
            print(f"  Change: {entropy_change:+.3f} {trend}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Analyze Expert Iteration Results')
    parser.add_argument('--results_dir', type=str, default='./ei_results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./ei_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--baseline_accuracy', type=float, default=0.0,
                        help='Baseline accuracy for comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading Expert Iteration experiment results...")
    experiments = analyze_all_experiments(args.results_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_validation_accuracy_curves(experiments, args.output_dir)
    plot_entropy_curves(experiments, args.output_dir)
    plot_generation_accuracy_curves(experiments, args.output_dir)
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary_df = generate_summary_table(experiments, args.output_dir)
    
    # Compare to baseline
    if args.baseline_accuracy > 0:
        compare_to_baseline(experiments, args.baseline_accuracy)
    
    # Analyze entropy trends
    analyze_entropy_trends(experiments)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print("Generated files:")
    print("- validation_accuracy_curves.png")
    print("- entropy_curves.png")
    print("- generation_accuracy_curves.png")
    print("- experiment_summary.csv")


if __name__ == '__main__':
    main()