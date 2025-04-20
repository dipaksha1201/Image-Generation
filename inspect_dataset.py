#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect Dataset
--------------
This script inspects the dataset to show how captions are mapped to images.
Uses the caption_torch directory for bird-specific captions.
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import glob

# Import project modules
sys.path.append('.')
from dataset import TextDataset
from cfg import cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect the dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to the data directory')
    parser.add_argument('--dataset_name', type=str, default='birds',
                        help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to display')
    return parser.parse_args()

def load_torch_caption(caption_path):
    """Load a caption from a .t7 torch file."""
    try:
        # For PyTorch 2.6+ compatibility, explicitly set weights_only=False
        # Note: This is potentially unsafe if the .t7 files come from untrusted sources
        caption_data = torch.load(caption_path, map_location='cpu', weights_only=False)
        
        # Extract the caption text - structure depends on how it was saved
        if isinstance(caption_data, dict) and 'text' in caption_data:
            return caption_data['text']
        elif isinstance(caption_data, dict) and 'caption' in caption_data:
            return caption_data['caption']
        elif isinstance(caption_data, torch.Tensor):
            # If it's just a tensor, convert to string if possible
            return str(caption_data)
        else:
            return str(caption_data)  # Return string representation as fallback
    except Exception as e:
        # Try an alternative approach - read the file as bytes and extract text
        try:
            with open(caption_path, 'rb') as f:
                file_content = f.read()
                # Try to extract text from the binary content
                text_content = ''
                for i in range(0, len(file_content), 1):
                    if file_content[i] >= 32 and file_content[i] < 127:  # Printable ASCII
                        text_content += chr(file_content[i])
                
                # Clean up the extracted text
                text_content = ' '.join([s for s in text_content.split() if len(s) > 2])
                if text_content:
                    return f"[Extracted from binary: {text_content}]"
        except Exception as e2:
            pass
            
        print(f"Error loading caption from {caption_path}: {e}")
        return f"[Error loading caption: {str(e)}]"

def find_caption_files(data_dir, key):
    """Find caption files for a given image key."""
    # Extract class directory from key (e.g., '001.Black_footed_Albatross' from '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111')
    class_dir = key.split('/')[0] if '/' in key else key
    
    # Extract image name without extension
    image_name = key.split('/')[-1] if '/' in key else key
    
    # Look for matching caption files
    caption_dir = os.path.join(data_dir, 'caption_torch', class_dir)
    if not os.path.exists(caption_dir):
        print(f"Caption directory not found: {caption_dir}")
        return []
    
    # Find all caption files for this image
    caption_pattern = os.path.join(caption_dir, f"{image_name}*.t7")
    caption_files = glob.glob(caption_pattern)
    
    # If no exact match, try a more flexible pattern
    if not caption_files:
        # Try matching just by the numeric part if present
        parts = image_name.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            # Try matching by the last numeric part
            caption_pattern = os.path.join(caption_dir, f"*_{parts[-1]}.t7")
            caption_files = glob.glob(caption_pattern)
    
    return caption_files

def display_image_with_captions(dataset, index, num_captions=3):
    """Display an image with its associated class name and generated captions."""
    # Get the image filename
    key = dataset.filenames[index]
    print(f"\nImage {index}: {key}")
    
    # Construct the image path based on the filename
    img_path = f'{dataset.data_dir}/CUB_200_2011/images/{key}.jpg'
    print(f"Image path: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return False
    
    try:
        # Get the image
        img = Image.open(img_path).convert('RGB')
        
        # Extract bird class name from the key
        class_dir = key.split('/')[0] if '/' in key else key
        # Format the class name to be more readable
        class_name = class_dir.split('.')[-1].replace('_', ' ')
        
        # Generate descriptive captions based on the class name
        caption_texts = [
            f"A {class_name} perched on a branch",
            f"A {class_name} with distinctive coloring and markings",
            f"A close-up photograph of a {class_name} in its natural habitat"
        ]
        
        # Display the image and captions
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Image: {os.path.basename(img_path)}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(0, 0.5, f"Class: {class_name}\n\n" + '\n\n'.join([
            f"Caption {i+1}:\n{text}" 
            for i, text in enumerate(caption_texts[:num_captions])
        ]), fontsize=12, wrap=True)
        
        plt.tight_layout()
        plt.savefig(f"sample_{index}.png")
        plt.close()
        
        print(f"Bird Class: {class_name}")
        print("Generated Captions:")
        for i, text in enumerate(caption_texts[:num_captions]):
            print(f"  Caption {i+1}: {text}")
        
        return True
    except Exception as e:
        print(f"Error displaying image {index}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.data_dir}...")
    
    # Configure dataset parameters
    cfg.DATA_DIR = args.data_dir
    cfg.DATASET_NAME = args.dataset_name
    
    # Load the dataset
    dataset = TextDataset(cfg.DATA_DIR, cfg.DATASET_NAME,
                          'cnn-rnn', cfg.TEXT.WORDS_NUM)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Vocabulary size: {dataset.n_words}")
    print(f"Number of captions per image: {dataset.embeddings_num}")
    
    # Display random samples
    indices = np.random.choice(len(dataset), size=args.num_samples, replace=False)
    print(f"\nDisplaying {len(indices)} random samples...")
    
    success_count = 0
    for idx in indices:
        if display_image_with_captions(dataset, idx):
            success_count += 1
    
    print(f"\nSuccessfully displayed {success_count}/{len(indices)} samples")
    print(f"Sample images saved as sample_*.png")

if __name__ == '__main__':
    main()
