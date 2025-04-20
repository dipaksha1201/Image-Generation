#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Generation Script
----------------------
This script provides a convenient way to generate images from text descriptions
using a trained model. It loads the trained models and generates images based on
provided text prompts.
"""

import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from cfg import cfg
from utils import cfg_from_file

# Define minimal versions of required functions from miscc modules
def mkdir_p(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


# Import model classes - you need to copy these from the original repo
from model import RNN_ENCODER, G_DCGAN, G_NET
from nltk.tokenize import RegexpTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate images from text descriptions')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1,
                        help='GPU device id to use [default: -1 for CPU]')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data',
                        help='Data directory containing captions and trained models')
    parser.add_argument('--net_g', dest='net_g', type=str, default='',
                        help='Path to generator model')
    parser.add_argument('--text_encoder', dest='text_encoder', type=str, default='',
                        help='Path to text encoder model')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='output',
                        help='Directory to save generated images')
    parser.add_argument('--prompts', dest='prompts', type=str, nargs='+',
                        help='Text descriptions to generate images from')
    parser.add_argument('--prompt_file', dest='prompt_file', type=str, default='',
                        help='File containing text descriptions (one per line)')
    args = parser.parse_args()
    return args


def prepare_prompts(args):
    """Prepare the text prompts for image generation."""
    prompts = []
    
    # Get prompts from command line arguments if provided
    if args.prompts:
        prompts.extend(args.prompts)
    
    # Get prompts from file if provided
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            file_prompts = [line.strip() for line in f.readlines() if line.strip()]
            prompts.extend(file_prompts)
    
    # Use default prompts if none provided
    if not prompts:
        prompts = [
            "A bird with a red head and a blue body",
            "A yellow bird with black wings",
            "A small bird with a orange belly and green wings",
            "A bird with a long beak"
        ]
        print("No prompts provided, using default examples.")
    
    return prompts


def load_text_encoder(text_encoder_path, device):
    """Load the text encoder model."""
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder not found at: {text_encoder_path}")
    
    # Load the trained text encoder
    state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
    vocab_size = state_dict['encoder.weight'].shape[0]
    print(f"Text encoder vocabulary size: {vocab_size}")
    
    # Initialize the text encoder
    text_encoder = RNN_ENCODER(vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)
    text_encoder.eval()
    
    return text_encoder, vocab_size


def load_generator(generator_path, device):
    """Load the generator model."""
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"Generator not found at: {generator_path}")
    
    # Initialize the generator based on configuration
    if cfg.GAN.B_DCGAN:
        netG = G_DCGAN()
    else:
        netG = G_NET()
    
    # Load the trained generator
    state_dict = torch.load(generator_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.to(device)
    netG.eval()
    
    print(f"Generator loaded from: {generator_path}")
    return netG


def load_word_dict(data_dir):
    """Load word dictionary for processing text."""
    captions_path = os.path.join(data_dir, 'captions.pickle')
    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Captions file not found at: {captions_path}")
        
    import pickle
    with open(captions_path, 'rb') as f:
        x = pickle.load(f)
        # Based on our inspection, ixtoword is at index 2 and wordtoix is at index 3
        ixtoword, wordtoix = x[2], x[3]
        print(f"Loaded dictionary with {len(wordtoix)} words")
        del x
    
    return wordtoix, ixtoword


def text_to_indices(text, wordtoix, max_words=18):
    """Convert text to indices for the text encoder."""
    # Tokenize the text
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    
    # If sentence is empty, use a default token
    if len(tokens) == 0:
        print(f"Warning: '{text}' contains no tokens. Using default.")
        tokens = ['<empty>']
    
    # Convert tokens to indices
    indices = []
    for token in tokens:
        if token in wordtoix:
            indices.append(wordtoix[token])
        else:
            # Skip unknown tokens
            print(f"Warning: '{token}' not in vocabulary, skipping.")
    
    # Ensure we have at least one token
    if len(indices) == 0:
        print(f"Warning: No known tokens in '{text}'. Using UNK token.")
        indices = [wordtoix.get('<unk>', 0)]
    
    # Truncate or pad to max_words
    if len(indices) > max_words:
        indices = indices[:max_words]
    
    caption_length = len(indices)
    
    # Pad with zeros if needed
    if len(indices) < max_words:
        indices.extend([0] * (max_words - len(indices)))
    
    return np.array(indices), caption_length


def generate_images(text_encoder, netG, device, prompts, wordtoix, ixtoword, output_dir):
    """Generate images from text descriptions."""
    batch_size = len(prompts)
    max_words = 18  # Default maximum words to use from each caption
    
    # Convert text prompts to indices
    captions = []
    cap_lens = []
    for prompt in prompts:
        caption, cap_len = text_to_indices(prompt, wordtoix, max_words)
        captions.append(caption)
        cap_lens.append(cap_len)
    
    # Convert to tensors
    captions = torch.LongTensor(np.array(captions)).to(device)
    cap_lens = torch.LongTensor(np.array(cap_lens)).to(device)
    
    # Generate noise for the generator
    noise = torch.FloatTensor(batch_size, cfg.GAN.Z_DIM).normal_(0, 1).to(device)
    
    # Generate images
    with torch.no_grad():
        # Generate text embeddings
        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
            
        # Generate images
        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        
        # Save the generated images
        for i in range(batch_size):
            # Save the highest resolution image (last one in the list)
            save_name = f"{i}_{prompts[i].replace(' ', '_')[:50]}"
            save_dir = os.path.join(output_dir, save_name)
            mkdir_p(save_dir)
            
            # Save images at different resolutions
            for j in range(len(fake_imgs)):
                img = fake_imgs[j][i].cpu().numpy()
                img = (img + 1.0) * 127.5  # Convert from [-1, 1] to [0, 255]
                img = img.astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))  # Change from CxHxW to HxWxC
                
                # Save the image
                im = Image.fromarray(img)
                fullpath = os.path.join(save_dir, f"gen_res{j}.png")
                im.save(fullpath)
                print(f"Saved image to {fullpath}")
            
            # Save attention maps if available
            if attention_maps is not None:
                for j in range(len(attention_maps)):
                    try:
                        attn_map = attention_maps[j][i].cpu().numpy()
                        
                        # Check the shape of the attention map
                        print(f"Attention map shape: {attn_map.shape}")
                        
                        # Handle different channel dimensions
                        if len(attn_map.shape) == 3:
                            # If it has 3 dimensions (C, H, W)
                            attn_map = np.transpose(attn_map, (1, 2, 0))  # CxHxW to HxWxC
                            
                            # If it has more than 3 channels, use only the first 3
                            if attn_map.shape[2] > 3:
                                print(f"Reducing attention map from {attn_map.shape[2]} to 3 channels")
                                attn_map = attn_map[:, :, :3]
                            # If it has 2 channels, add a third one
                            elif attn_map.shape[2] == 2:
                                print(f"Expanding attention map from 2 to 3 channels")
                                zeros = np.zeros((attn_map.shape[0], attn_map.shape[1], 1))
                                attn_map = np.concatenate([attn_map, zeros], axis=2)
                            # If it has 1 channel, convert to grayscale (1 channel)
                            elif attn_map.shape[2] == 1:
                                print(f"Converting single-channel attention map to grayscale")
                                attn_map = attn_map[:, :, 0]
                        
                        # Ensure values are in valid range and convert to uint8
                        attn_map = np.clip(attn_map * 255, 0, 255).astype(np.uint8)
                        
                        # Create and save the image
                        attn_im = Image.fromarray(attn_map)
                        fullpath = os.path.join(save_dir, f"attn_map{j}.png")
                        attn_im.save(fullpath)
                        print(f"Saved attention map to {fullpath}")
                    except Exception as e:
                        print(f"Error processing attention map {j}: {e}")


def main():
    """Main function to run image generation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    if args.cfg_file != '':
        cfg_from_file(args.cfg_file)
    
    # Set up device
    if args.gpu_id != -1:
        # Check for Apple Silicon (MPS) support
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU via MPS")
        # Check for CUDA support
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id}")
            print(f"Using CUDA GPU device {args.gpu_id}")
            cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            print("GPU requested but neither MPS nor CUDA is available. Using CPU device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Create output directory
    output_dir = args.output_dir
    mkdir_p(output_dir)
    
    # Determine paths
    data_dir = args.data_dir
    netG_path = args.net_g if args.net_g else os.path.join('Model', 'bird_AttnGAN2.pth')
    text_encoder_path = args.text_encoder if args.text_encoder else os.path.join('DAMSMencoders/bird', 'text_encoder200.pth')
    
    # Load models
    text_encoder, vocab_size = load_text_encoder(text_encoder_path, device)
    netG = load_generator(netG_path, device)
    
    # Load word dictionary
    try:
        wordtoix, ixtoword = load_word_dict(data_dir)
    except FileNotFoundError:
        print("Word dictionary not found. Using dummy dictionary for testing.")
        wordtoix = {'a': 1, 'the': 2, 'bird': 3, 'with': 4, 'red': 5, 'blue': 6, 'head': 7, 'body': 8}
        ixtoword = {v: k for k, v in wordtoix.items()}
    
    # Get text prompts
    prompts = prepare_prompts(args)
    print(f"Generating images for {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")
    
    # Generate images
    generate_images(text_encoder, netG, device, prompts, wordtoix, ixtoword, output_dir)
    print(f"All images generated successfully in {output_dir}")


if __name__ == '__main__':
    main()
