import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
from cfg import cfg
from dataset import TextDataset
from trainer import condGANTrainer 
from trainer_contrastive import ContrastiveGANTrainer
import argparse
import numpy as np
from torch.utils.data import DataLoader
from contrastive_encoders import MateBatchSampler


def load_data(split='train', shuffle=True, use_mate_sampler=False):
    """
    Returns (dataloader, dataset).

    If `use_mate_sampler` is True the loader will yield 2×BATCH_SIZE
    items per iteration: N anchors + N mates.
    """
    # --- dataset stays exactly the same ----------------------------------
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()
    ])

    dataset = TextDataset(cfg.DATA_DIR, split,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    # --- diagnostics (unchanged) ----------------------------------------
    print(f"n_words: {dataset.n_words}")
    print(f"embeddings_num: {dataset.embeddings_num}")
    print(f"# images (filenames): {len(dataset.filenames)}")
    print(f"# caption sentences : {len(dataset.captions)}")
    print("First 3 image keys  :", dataset.filenames[:3])

    # --- choose loader strategy -----------------------------------------
    if use_mate_sampler:
        # N anchors per batch  → loader yields 2N samples
        anchor_bs   = cfg.TRAIN.BATCH_SIZE
        batch_sampler = MateBatchSampler(dataset, batch_size=anchor_bs)
        dataloader = DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=int(cfg.WORKERS))
    else:
        # classic loader (same as before)
        dataloader = DataLoader(dataset,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                shuffle=shuffle,
                                drop_last=True,
                                num_workers=int(cfg.WORKERS))

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
        # Load data with detailed debugging
        print("Loading data...")
        try:
            if args.mode == 'train':
                print("Loading training data with use_mate_sampler=True")
                dataloader, dataset = load_data(split='train', shuffle=True, use_mate_sampler=True)
            else:
                print("Loading test data with use_mate_sampler=True")
                dataloader, dataset = load_data(split='test', shuffle=False, use_mate_sampler=True)
            print(f"Successfully loaded dataset with {len(dataset)} samples")
            print(f"Dataloader has {len(dataloader)} batches")
            print(f"Dataset type: {type(dataset)}, Dataloader type: {type(dataloader)}")
            
            # Test iteration through dataloader
            print("Testing dataloader iteration...")
            data_iter = iter(dataloader)
            print("Getting first batch...")
            first_batch = next(data_iter)
            print(f"First batch type: {type(first_batch)}")
            if isinstance(first_batch, dict):
                print(f"First batch keys: {first_batch.keys()}")
            elif isinstance(first_batch, (list, tuple)):
                print(f"First batch length: {len(first_batch)}")
                print(f"First batch element types: {[type(item) for item in first_batch]}")
        except Exception as e:
            print(f"Error during data loading/validation: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Initialize trainer
        print("Initializing trainer...")
        try:
            print(f"Creating trainer with dataset.n_words={dataset.n_words}")
            trainer = ContrastiveGANTrainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
            print("Trainer initialized successfully")
        except Exception as e:
            print(f"Error during trainer initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
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
