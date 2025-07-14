#!/usr/bin/env python3
"""
Script to download the required models for the supplemental assignment.
"""

import os
from huggingface_hub import snapshot_download

def download_llama_models():
    """Download required Llama models for the supplemental assignment."""
    
    models = [
        {
            "name": "Llama-3.1-8B",
            "repo_id": "meta-llama/Llama-3.1-8B",
            "local_dir": "./models/Llama-3.1-8B"
        },
        {
            "name": "Llama-3.3-70B-Instruct", 
            "repo_id": "meta-llama/Llama-3.3-70B-Instruct",
            "local_dir": "./models/Llama-3.3-70B-Instruct"
        }
    ]
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    for model in models:
        print(f"\nDownloading {model['name']} to {model['local_dir']}...")
        print("This may take a while depending on your internet connection...")
        
        try:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=model["local_dir"],
                local_dir_use_symlinks=False,
                resume_download=True,
                token=True  # Use HF token if available
            )
            print(f"Successfully downloaded {model['name']} to {model['local_dir']}")
            
        except Exception as e:
            print(f"Error downloading {model['name']}: {e}")
            print(f"\nAlternative download methods:")
            print(f"1. Using huggingface-cli:")
            print(f"   huggingface-cli download {model['repo_id']} --local-dir {model['local_dir']}")
            print(f"2. Using git:")
            print(f"   git clone https://huggingface.co/{model['repo_id']} {model['local_dir']}")
            print(f"Note: You may need to request access to {model['repo_id']} on HuggingFace")

if __name__ == "__main__":
    download_llama_models()