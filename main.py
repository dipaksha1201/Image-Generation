import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from cfg import cfg
from dataset import TextDataset
from trainer import condGANTrainer
import argparse
import numpy as np


def load_data():
    """
    Load data using dataset.py.
    Returns a dataloader and dataset information.
    """
    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    
    print("Dataset info:")
    print(f"n_words: {dataset.n_words}")
    print(f"embeddings_num: {dataset.embeddings_num}")
    print(f"Number of filenames: {len(dataset.filenames)}")
    print(f"Number of captions: {len(dataset.captions)}")
    print(f"First few filenames: {dataset.filenames[:3] if dataset.filenames else 'None'}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    
    return dataloader, dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train or sample from a Text-to-Image model')
    # Removed cfg_file argument since we're using cfg.py directly
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to use')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='output',
                        help='Directory to save generated images')
    parser.add_argument('--mode', dest='mode', type=str, default='sample',
                        help='train or sample')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def main():
    """
    Main function to load data and perform sampling using trainer.
    """
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility if provided
    if args.manualSeed is not None:
        random_seed = args.manualSeed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have a seed_all function, but we set the device seed
            torch.mps.manual_seed(random_seed)
        elif torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    
    # Skip loading YAML config file - using cfg.py directly instead
    print("Using configuration from cfg.py directly")
    
    # Update config from command line arguments
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
        
    # Update model paths based on command line arguments
    print("Updated model paths:")
    print(f"Generator: {cfg.TRAIN.NET_G}")
    print(f"Text Encoder: {cfg.TRAIN.NET_E}")
        
    # Check for Apple Silicon (MPS) support
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        print("Apple Silicon detected - will use MPS backend")
        cfg.CUDA = True  # We'll treat MPS as a CUDA-like device
    elif not torch.cuda.is_available() and cfg.CUDA:
        print("CUDA requested but not available. Setting CUDA to False.")
        cfg.CUDA = False
    
    # Make sure WORKERS is set in cfg
    if not hasattr(cfg, 'WORKERS'):
        cfg.WORKERS = 4
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Load data
        print("Loading data...")
        dataloader, dataset = load_data()
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        print(f"Dataloader has {len(dataloader)} batches")
        
        # Initialize trainer
        print("Initializing trainer...")
        trainer = condGANTrainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
        
        # Train or sample based on mode
        if args.mode == 'train':
            print("Starting training...")
            trainer.train()
        else:  # sample mode
            print("Starting sampling...")
            trainer.sampling('test')
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
