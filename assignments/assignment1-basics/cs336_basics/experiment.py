"""
Experiment tracking and management module.

This module provides tools for managing and recording deep learning experiments,
tracking validation and training losses, and recording experimental configurations
and results. Supports both local logging and Weights & Biases integration.
"""
import os
import time
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Experiment tracker for recording and managing training experiments.
    
    Supports both local file and Weights & Biases recording modes.
    """
    
    def __init__(
        self, 
        experiment_name: str,
        base_dir: str = "./experiments",
        use_wandb: bool = False,
        wandb_project: str = "cs336-experiments",
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        resume: bool = False,
        wandb_id: Optional[str] = None,
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for saving experiment data
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (organization or username)
            config: Experiment configuration parameters
            tags: Experiment tags
            resume: Whether to resume a previous experiment
            wandb_id: W&B run ID to resume
        """
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "steps_per_sec": [],
            "gradient_steps": [],
            "wallclock_times": [],
        }
        
        # Initialize experiment directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Record configuration
        self.config = config or {}
        self.config["experiment_name"] = experiment_name
        self.config["timestamp"] = timestamp
        
        # Save configuration to JSON
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("Weights & Biases (wandb) not installed. Using local logging only.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config,
                    tags=tags,
                    resume="allow" if resume else None,
                    id=wandb_id
                )
                
                # Save wandb run ID for resuming
                with open(self.experiment_dir / "wandb_id.txt", "w") as f:
                    f.write(wandb.run.id)
    
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step number
        """
        # Calculate wall clock time
        wallclock_time = time.time() - self.start_time
        
        # Update internal metrics storage
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Record step and time
        self.metrics["gradient_steps"].append(step)
        self.metrics["wallclock_times"].append(wallclock_time)
        
        # Write to JSONL format log file
        log_entry = {
            "step": step,
            "wallclock_time": wallclock_time,
            **metrics
        }
        
        with open(self.experiment_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to wandb (if enabled)
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log experiment summary metrics.
        
        Args:
            summary: Summary metrics to log
        """
        # Save summary
        with open(self.experiment_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Log to wandb (if enabled)
        if self.use_wandb:
            for key, value in summary.items():
                wandb.run.summary[key] = value
    
    def save_artifact(self, artifact_path: Union[str, os.PathLike], name: str) -> None:
        """
        Save experiment artifact (such as model checkpoint).
        
        Args:
            artifact_path: Path to the artifact
            name: Name of the artifact
        """
        # Copy artifact to experiment directory
        artifact_path = Path(artifact_path)
        dest_path = self.experiment_dir / "artifacts" / artifact_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If it's a file, copy it
        if artifact_path.is_file():
            import shutil
            shutil.copy2(artifact_path, dest_path)
        
        # Log to wandb (if enabled)
        if self.use_wandb:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
    
    def plot_metrics(self, save_path: Optional[Union[str, os.PathLike]] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.metrics["train_loss"]:
            logger.warning("No metrics to plot.")
            return
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot loss vs gradient steps
        axs[0].plot(self.metrics["gradient_steps"], self.metrics["train_loss"], label="Training Loss")
        if self.metrics["val_loss"]:
            axs[0].plot(self.metrics["gradient_steps"], self.metrics["val_loss"], label="Validation Loss")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Loss vs Gradient Steps")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot loss vs wall clock time
        axs[1].plot(self.metrics["wallclock_times"], self.metrics["train_loss"], label="Training Loss")
        if self.metrics["val_loss"]:
            axs[1].plot(self.metrics["wallclock_times"], self.metrics["val_loss"], label="Validation Loss")
        axs[1].set_xlabel("Time (seconds)")
        axs[1].set_ylabel("Loss")
        axs[1].set_title("Loss vs Wall Clock Time")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.experiment_dir / "loss_curves.png")
        
        # Close the figure
        plt.close()
    
    def finish(self) -> None:
        """
        Finish experiment recording.
        """
        # Calculate total training time
        total_time = time.time() - self.start_time
        
        # Record summary
        summary = {
            "total_training_time": total_time,
            "total_steps": len(self.metrics["gradient_steps"]),
            "final_train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else None,
            "final_val_loss": self.metrics["val_loss"][-1] if self.metrics["val_loss"] else None,
            "min_val_loss": min(self.metrics["val_loss"]) if self.metrics["val_loss"] else None,
        }
        
        self.log_summary(summary)
        
        # Plot and save metrics
        self.plot_metrics()
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()
        
        logger.info(f"Experiment completed! Results saved in {self.experiment_dir}")


def create_experiment_log(
    experiment_list: List[Dict[str, Any]],
    output_path: Union[str, os.PathLike] = "experiment_log.md"
) -> None:
    """
    Create experiment log document.
    
    Args:
        experiment_list: List of experiments, each as a dictionary
        output_path: Output path
    """
    output_path = Path(output_path)
    
    with open(output_path, "w") as f:
        f.write("# Experiment Log\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Summary\n\n")
        f.write("| Experiment Name | Configuration | Best Val Loss | Training Time |\n")
        f.write("|----------------|--------------|--------------|---------------|\n")
        
        for exp in experiment_list:
            name = exp.get("name", "Unnamed")
            config = str(exp.get("config", {})).replace("\n", " ")
            best_val_loss = exp.get("best_val_loss", "N/A")
            training_time = exp.get("training_time", "N/A")
            
            f.write(f"| {name} | {config} | {best_val_loss} | {training_time} |\n")
        
        f.write("\n## Detailed Experiments\n\n")
        
        for i, exp in enumerate(experiment_list):
            f.write(f"### Experiment {i+1}: {exp.get('name', 'Unnamed')}\n\n")
            
            # Configuration
            f.write("#### Configuration:\n\n")
            f.write("```python\n")
            f.write(json.dumps(exp.get("config", {}), indent=2))
            f.write("\n```\n\n")
            
            # Results
            f.write("#### Results:\n\n")
            f.write(f"- Best validation loss: {exp.get('best_val_loss', 'N/A')}\n")
            f.write(f"- Training time: {exp.get('training_time', 'N/A')}\n")
            
            # Observations and conclusions
            if "observations" in exp:
                f.write("\n#### Observations and Conclusions:\n\n")
                f.write(exp["observations"])
                f.write("\n\n")
            
            # Plot path
            if "plot_path" in exp:
                f.write(f"![Loss curves]({exp['plot_path']})\n\n")
            
            f.write("---\n\n")
        
        f.write("## Ablation Studies\n\n")
        f.write("Describe ablation studies here, comparing the effects of different components or hyperparameters.\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("Summarize key findings from the experiments here.\n")
    
    logger.info(f"Experiment log saved to {output_path}")


def load_experiment_from_dir(experiment_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load experiment data from directory.
    
    Args:
        experiment_dir: Experiment directory
        
    Returns:
        Dictionary containing experiment data
    """
    experiment_dir = Path(experiment_dir)
    
    # Load configuration
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        logger.warning(f"No config.json found in {experiment_dir}")
        config = {}
    else:
        with open(config_path) as f:
            config = json.load(f)
    
    # Load metrics
    metrics_path = experiment_dir / "metrics.jsonl"
    metrics = {
        "gradient_steps": [],
        "wallclock_times": [],
        "train_loss": [],
        "val_loss": [],
    }
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                data = json.loads(line)
                for key in metrics:
                    if key == "gradient_steps":
                        metrics[key].append(data.get("step", 0))
                    elif key == "wallclock_times":
                        metrics[key].append(data.get("wallclock_time", 0))
                    else:
                        if key in data:
                            metrics[key].append(data[key])
    
    # Load summary
    summary_path = experiment_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Return experiment data
    return {
        "name": config.get("experiment_name", "Unknown"),
        "config": config,
        "metrics": metrics,
        "summary": summary,
        "dir": experiment_dir,
    }


def compare_experiments(
    experiment_dirs: List[Union[str, os.PathLike]],
    metrics: List[str] = ["train_loss", "val_loss"],
    save_path: Optional[Union[str, os.PathLike]] = None
) -> None:
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directories
        metrics: Metrics to compare
        save_path: Path to save the plot
    """
    experiments = [load_experiment_from_dir(d) for d in experiment_dirs]
    
    # Create figure
    fig, axs = plt.subplots(len(metrics), 2, figsize=(15, 5 * len(metrics)))
    
    for i, metric in enumerate(metrics):
        # Step comparison plot
        for exp in experiments:
            if metric in exp["metrics"] and exp["metrics"][metric] and "gradient_steps" in exp["metrics"]:
                axs[i, 0].plot(
                    exp["metrics"]["gradient_steps"],
                    exp["metrics"][metric],
                    label=exp["name"]
                )
        
        axs[i, 0].set_xlabel("Gradient Steps")
        axs[i, 0].set_ylabel(metric)
        axs[i, 0].set_title(f"{metric} vs Gradient Steps")
        axs[i, 0].legend()
        axs[i, 0].grid(True)
        
        # Time comparison plot
        for exp in experiments:
            if metric in exp["metrics"] and exp["metrics"][metric] and "wallclock_times" in exp["metrics"]:
                axs[i, 1].plot(
                    exp["metrics"]["wallclock_times"],
                    exp["metrics"][metric],
                    label=exp["name"]
                )
        
        axs[i, 1].set_xlabel("Time (seconds)")
        axs[i, 1].set_ylabel(metric)
        axs[i, 1].set_title(f"{metric} vs Time")
        axs[i, 1].legend()
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig("experiment_comparison.png")
    
    plt.close()