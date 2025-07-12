"""
SFT (Supervised Fine-tuning) implementation for MATH dataset.
Complete implementation of all required functions for Chapter 4.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Any, Optional, Tuple
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset


class SFTDataset(Dataset):
    """Dataset class for SFT data."""
    
    def __init__(self, dataset_name: str, tokenizer, prompt_template: str, split: str = "train"):
        """
        Initialize SFT dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            tokenizer: Tokenizer to use
            prompt_template: Template for formatting prompts
            split: The dataset split to use (e.g., 'train', 'test')
        """
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.data = self.load_data(dataset_name, split)
    
    def load_data(self, dataset_name: str, split: str) -> List[Dict[str, Any]]:
        """Load data from HuggingFace Hub."""
        print(f"Loading dataset {dataset_name} from HuggingFace Hub...")
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(dataset)} examples from {dataset_name}")
        return dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # The Countdown dataset has 'question' and 'solution' columns.
        # Let's adjust this to be more general if needed, but for now this works.
        problem = item.get('question', item.get('problem', ''))
        solution = item.get('solution', '')
        
        # Format prompt using template
        prompt = self.prompt_template.format(question=problem)
        
        return {
            'prompt': prompt,
            'output': solution,
            'problem': problem,
            'solution': solution
        }