#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect Pickle File
------------------
This script inspects the contents of a pickle file to understand its structure.
"""

import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect a pickle file')
    parser.add_argument('--pickle_file', type=str, default='data/captions.pickle',
                        help='Path to the pickle file to inspect')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to print')
    return parser.parse_args()

def inspect_pickle(pickle_file, num_samples=10):
    """Inspect the contents of a pickle file."""
    if not os.path.exists(pickle_file):
        print(f"Error: File not found at {pickle_file}")
        return
    
    print(f"Loading pickle file: {pickle_file}")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type of loaded data: {type(data)}")
    
    # Special handling for captions.pickle which has a specific format
    if pickle_file.endswith('captions.pickle') and isinstance(data, list) and len(data) >= 4:
        train_captions, test_captions = data[0], data[1]
        ixtoword, wordtoix = data[2], data[3]
        
        print("\n=== CAPTIONS PICKLE ANALYSIS ===")
        print(f"Train captions: {len(train_captions)} items")
        print(f"Test captions: {len(test_captions)} items")
        print(f"Vocabulary size: {len(ixtoword)} words")
        
        # Sample captions
        print("\nSample train captions:")
        for i, caption in enumerate(train_captions[:num_samples]):
            print(f"  Caption {i}: {caption}")
            # If caption is a list of indices, convert to words
            if isinstance(caption, list) and all(isinstance(idx, int) for idx in caption):
                words = [ixtoword.get(idx, '[UNK]') for idx in caption]
                print(f"    Decoded: {' '.join(words)}")
        
        print("\nSample test captions:")
        for i, caption in enumerate(test_captions[:num_samples]):
            print(f"  Caption {i}: {caption}")
            # If caption is a list of indices, convert to words
            if isinstance(caption, list) and all(isinstance(idx, int) for idx in caption):
                words = [ixtoword.get(idx, '[UNK]') for idx in caption]
                print(f"    Decoded: {' '.join(words)}")
        
        # Sample vocabulary
        print("\nSample vocabulary (index -> word):")
        sample_indices = list(ixtoword.keys())[:min(num_samples, len(ixtoword))]
        for idx in sample_indices:
            print(f"  {idx}: {ixtoword[idx]}")
        
        return
    
    # General inspection for other pickle files
    if isinstance(data, list):
        print(f"List length: {len(data)}")
        for i, item in enumerate(data):
            if i >= num_samples:
                print(f"... (showing {num_samples} of {len(data)} items)")
                break
            print(f"\nItem {i}:")
            print(f"  Type: {type(item)}")
            if isinstance(item, dict):
                print(f"  Dictionary with {len(item)} keys")
                print(f"  Sample keys: {list(item.keys())[:5]}")
                # Print a few sample values
                for j, (key, value) in enumerate(item.items()):
                    if j >= 5:
                        break
                    print(f"    {key}: {value}")
            elif isinstance(item, list):
                print(f"  List with {len(item)} items")
                # Print a few sample items
                print(f"  Sample items: {item[:min(num_samples, len(item))]}")
            else:
                print(f"  Value: {item}")
    elif isinstance(data, dict):
        print(f"Dictionary with {len(data)} keys")
        print(f"Keys: {list(data.keys())}")
        # Print a few sample values
        for i, (key, value) in enumerate(data.items()):
            if i >= num_samples:
                print(f"... (showing {num_samples} of {len(data)} keys)")
                break
            print(f"\nKey: {key}")
            print(f"  Type: {type(value)}")
            if isinstance(value, dict):
                print(f"  Dictionary with {len(value)} keys")
                print(f"  Sample keys: {list(value.keys())[:5]}")
            elif isinstance(value, list):
                print(f"  List with {len(value)} items")
                print(f"  Sample items: {value[:min(5, len(value))]}")
            else:
                print(f"  Value: {value}")
    else:
        print(f"Data is of type {type(data)}")
        print(f"Value: {data}")

def main():
    args = parse_args()
    inspect_pickle(args.pickle_file, args.num_samples)

if __name__ == '__main__':
    main()
