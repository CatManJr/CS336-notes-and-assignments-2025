#!/usr/bin/env python3
"""
Script to download the Qwen 2.5 Math 1.5B model locally.
"""

import os
from huggingface_hub import snapshot_download

def download_model():
    """Download Qwen 2.5 Math 1.5B model to local directory."""
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    local_dir = "./models/Qwen2.5-Math-1.5B"
    
    print(f"Downloading {model_name} to {local_dir}...")
    print("This may take a while depending on your internet connection...")
    
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Successfully downloaded model to {local_dir}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                print(f"  {os.path.join(root, file)}")
                
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternative download methods:")
        print("1. Using huggingface-cli:")
        print(f"   huggingface-cli download {model_name} --local-dir {local_dir}")
        print("2. Using git:")
        print(f"   git clone https://huggingface.co/{model_name} {local_dir}")

if __name__ == "__main__":
    download_model()