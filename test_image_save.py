#!/usr/bin/env python
# Test script to diagnose the "cannot open resource" error in image saving

import os
import torch
import numpy as np
from PIL import Image
from cfg import cfg
from utils import mkdir_p, build_super_images
from model import G_NET, RNN_ENCODER, CNN_ENCODER
from GlobalAttention import func_attention
from losses import words_loss

def test_image_save():
    """Test the image saving functionality with controlled inputs"""
    print("Starting image save test...")
    
    # Create test directories
    output_dir = 'test_output'
    image_dir = os.path.join(output_dir, 'Image')
    model_dir = os.path.join(output_dir, 'Model')
    
    print(f"Creating test directories: {output_dir}, {image_dir}, {model_dir}")
    mkdir_p(output_dir)
    mkdir_p(image_dir)
    mkdir_p(model_dir)
    
    # Check if MPS is available (Apple Silicon) or fall back to CUDA or CPU
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU via MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using CUDA device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    # Create a simple test image
    print("Creating test image...")
    try:
        # Create a simple RGB test image (3 channels)
        test_img = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add some color to make it more interesting
        test_img[10:30, 10:30, 0] = 255  # Red square
        test_img[30:50, 30:50, 1] = 255  # Green square
        test_img[20:40, 30:50, 2] = 255  # Blue square
        
        # Test saving directly with PIL
        print("Testing direct PIL image save...")
        try:
            im = Image.fromarray(test_img)
            print(f"PIL Image created with size: {im.size}, mode: {im.mode}")
            fullpath = os.path.join(image_dir, 'test_direct.png')
            print(f"Saving image to: {fullpath}")
            im.save(fullpath)
            print(f"Successfully saved direct image to: {fullpath}")
        except Exception as e:
            print(f"Error in direct image save: {e}")
            import traceback
            traceback.print_exc()
        
        # Test saving with build_super_images
        print("\nTesting build_super_images...")
        try:
            # Create dummy inputs for build_super_images
            batch_size = 8
            # Create fake images tensor (batch_size, channels, height, width)
            fake_imgs = torch.zeros(batch_size, 3, 64, 64, device=device)
            # Create fake captions
            captions = torch.zeros(batch_size, 18, device=device).long()
            # Create fake ixtoword dictionary
            ixtoword = {i: f'word_{i}' for i in range(100)}
            # Create fake attention maps
            att_maps = []
            for i in range(batch_size):
                att_map = torch.zeros(18, 17, 17, device=device)
                # Add some pattern to attention maps
                for j in range(18):
                    att_map[j, j % 17, j % 17] = 1.0
                att_maps.append(att_map.unsqueeze(0))
            
            # Move tensors to CPU for build_super_images
            fake_imgs_cpu = fake_imgs.detach().cpu()
            
            print(f"Inputs prepared: fake_imgs shape={fake_imgs_cpu.shape}, len(att_maps)={len(att_maps)}")
            print(f"Calling build_super_images...")
            
            img_set, _ = build_super_images(fake_imgs_cpu, captions, ixtoword, att_maps, 17)
            
            if img_set is not None:
                print(f"Super image created with shape: {img_set.shape}, dtype: {img_set.dtype}")
                try:
                    im = Image.fromarray(img_set)
                    print(f"PIL Image created with size: {im.size}, mode: {im.mode}")
                    
                    fullpath = os.path.join(image_dir, 'test_super.png')
                    print(f"Saving super image to: {fullpath}")
                    im.save(fullpath)
                    print(f"Successfully saved super image to: {fullpath}")
                except Exception as e:
                    print(f"Error creating/saving super image: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("img_set is None, skipping super image creation")
        
        except Exception as e:
            print(f"Error in build_super_images test: {e}")
            import traceback
            traceback.print_exc()
            
        print("\nTest completed.")
        
    except Exception as e:
        print(f"Error in test_image_save: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_save()
