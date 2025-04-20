import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm
from scipy import linalg

def load_images_from_prompt_dirs(output_dir, image_name='gen_res2.png'):
    """
    Load all gen_res2.png images from each prompt directory in the output folder.
    
    Args:
        output_dir (str): Path to the output directory containing prompt folders
        image_name (str): Name of the image file to load (default: 'gen_res2.png')
        
    Returns:
        torch.Tensor: Batch of images normalized to [0, 1] with shape [N, 3, H, W]
    """
    # Get all prompt directories
    prompt_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    if not prompt_dirs:
        raise ValueError(f"No prompt directories found in {output_dir}")
    
    # Prepare image transformation - normalize for Inception model
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception v3 expects 299x299 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    images = []
    print(f"Loading {image_name} images from {len(prompt_dirs)} prompt directories...")
    
    for prompt_dir in tqdm(prompt_dirs):
        img_path = os.path.join(output_dir, prompt_dir, image_name)
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        else:
            print(f"Warning: {img_path} not found")
    
    if not images:
        raise ValueError(f"No valid {image_name} images found in prompt directories")
    
    # Stack all images into a single tensor
    return torch.stack(images)

def calculate_inception_score(images, device='mps', splits=1):
    """
    Calculate the Inception Score using torchvision's Inception model.
    
    Args:
        images (torch.Tensor): Batch of images with shape [N, 3, H, W] normalized for Inception
        device (str): Device to run the model on ('mps', 'cuda', or 'cpu')
        splits (int): Number of splits to use for calculating IS
        
    Returns:
        tuple: (mean_inception_score, std_inception_score)
    """
    # Check device availability - prioritize Metal on macOS
    if device == 'mps' and torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS) for acceleration")
    elif device == 'cuda' and torch.cuda.is_available():
        print("Using CUDA for acceleration")
    else:
        device = 'cpu'
        print("GPU acceleration not available, using CPU instead")
    
    device = torch.device(device)
    
    # Load pre-trained Inception v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # Remove the classification layer
    inception_model.eval()
    inception_model.to(device)
    
    # Move images to device
    images = images.to(device)
    
    # Get predictions in batches to avoid memory issues
    batch_size = 32
    n_batches = int(np.ceil(len(images) / batch_size))
    
    preds = []
    with torch.no_grad():
        for i in range(n_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]
            pred = inception_model(batch)
            # Convert features to probabilities
            pred = F.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate Inception Score
    scores = []
    n_images = preds.shape[0]
    
    # Adjust splits if we have too few images
    if n_images < splits:
        print(f"Warning: Too few images ({n_images}) for {splits} splits. Setting splits to 1.")
        splits = 1
    
    n_part = n_images // splits
    
    for i in range(splits):
        part = preds[i * n_part:(i + 1) * n_part, :]
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-10
        part = np.maximum(part, epsilon)
        mean_part = np.maximum(np.mean(part, axis=0, keepdims=True), epsilon)
        kl = part * (np.log(part) - np.log(mean_part))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    # Load all gen_res2.png images
    images = load_images_from_prompt_dirs(output_dir, 'gen_res2.png')
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    
    # Calculate Inception Score
    print("Calculating Inception Score...")
    # Prioritize Metal on macOS, then CUDA, then fall back to CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    IS, IS_std = calculate_inception_score(images, device=device)
    print(f"Inception Score: {IS:.4f} ± {IS_std:.4f}")

if __name__ == "__main__":
    main()