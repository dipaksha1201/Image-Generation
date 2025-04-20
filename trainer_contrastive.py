from __future__ import print_function
# Replace six.moves.range with built-in range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from PIL import Image

from cfg import cfg
from utils import mkdir_p
from utils import build_super_images, build_super_images2
from utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from dataset import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from contrastive_encoders import prepare_contrastive_data
import torch.nn.functional as F
from contrastive_encoders import build_pos_mask, sup_con_loss

from losses import words_loss
from losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys

# ################# Text to image task############################ #
class ContrastiveGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        # Check if MPS is available (Apple Silicon) or fall back to CUDA or CPU
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU via MPS")
        elif torch.cuda.is_available() and cfg.CUDA:
            self.device = torch.device(f'cuda:{cfg.GPU_ID}')
            torch.cuda.set_device(cfg.GPU_ID)
            cudnn.benchmark = True
            print(f"Using CUDA device {cfg.GPU_ID}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU device")

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            try:
                # Try to extract epoch number from filename (e.g., model_100.pth -> 100)
                istart = cfg.TRAIN.NET_G.rfind('_') + 1
                iend = cfg.TRAIN.NET_G.rfind('.')
                epoch_str = cfg.TRAIN.NET_G[istart:iend]
                epoch = int(epoch_str) + 1
                print(f"Resuming from epoch {epoch}")
            except ValueError:
                # If filename doesn't contain a valid epoch number, start from epoch 1
                print(f"Could not extract epoch number from {cfg.TRAIN.NET_G}, starting from epoch 1")
                epoch = 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    try:
                        # First try to find discriminator in the same directory as generator
                        s_tmp = Gname[:Gname.rfind('/')]
                        Dname = '%s/netD%d.pth' % (s_tmp, i)
                        print('Looking for D at: ', Dname)
                        state_dict = \
                            torch.load(Dname, map_location=lambda storage, loc: storage)
                        netsD[i].load_state_dict(state_dict)
                        print(f"Loaded discriminator {i} from {Dname}")
                    except Exception as e:
                        # If not found, try to find it in the Model directory
                        try:
                            Dname = f"Model/netD{i}.pth"
                            print(f"Looking for D at: {Dname}")
                            state_dict = \
                                torch.load(Dname, map_location=lambda storage, loc: storage)
                            netsD[i].load_state_dict(state_dict)
                            print(f"Loaded discriminator {i} from {Dname}")
                        except Exception as e2:
                            print(f"Error loading discriminator {i}: {e2}")
                            print(f"Initializing discriminator {i} with random weights")
                    except Exception as e:
                        print(f'Error loading discriminator model {Dname}: {e}')
                        print('Using randomly initialized weights for this discriminator.')
        # ########################################################### #
        # Move models to the appropriate device
        text_encoder = text_encoder.to(self.device)
        image_encoder = image_encoder.to(self.device)
        netG = netG.to(self.device)
        for i in range(len(netsD)):
            netsD[i] = netsD[i].to(self.device)
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        # For contrastive training with mate sampler, we need double the batch size
        batch_size = self.batch_size 
        full_batch_size = batch_size * 2
        
        # Create labels for the full batch size
        real_labels = torch.FloatTensor(full_batch_size).fill_(1).to(self.device)
        fake_labels = torch.FloatTensor(full_batch_size).fill_(0).to(self.device)
        match_labels = torch.LongTensor(range(full_batch_size)).to(self.device)
        
        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        try:
            print(f"\n----- DEBUG: save_img_results -----")
            print(f"Image directory: {self.image_dir}")
            print(f"Directory exists: {os.path.exists(self.image_dir)}")
            
            # Ensure image directory exists
            if not os.path.exists(self.image_dir):
                print(f"Creating image directory: {self.image_dir}")
                mkdir_p(self.image_dir)
                
            # Save images
            print(f"Generating fake images...")
            fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
            print(f"Generated {len(fake_imgs)} fake images with {len(attention_maps)} attention maps")
            
            for i in range(len(attention_maps)):
                print(f"Processing attention map {i}...")
                if len(fake_imgs) > 1:
                    img = fake_imgs[i + 1].detach().cpu()
                    lr_img = fake_imgs[i].detach().cpu()
                    print(f"Using multi-resolution images: img shape={img.shape}, lr_img shape={lr_img.shape}")
                else:
                    img = fake_imgs[0].detach().cpu()
                    lr_img = None
                    print(f"Using single resolution image: img shape={img.shape}")
                    
                attn_maps = attention_maps[i]
                att_sze = attn_maps.size(2)
                print(f"Attention map size: {att_sze}")
                
                print(f"Building super images...")
                img_set, _ = \
                    build_super_images(img, captions, self.ixtoword,
                                       attn_maps, att_sze, lr_imgs=lr_img)
                                       
                if img_set is not None:
                    print(f"Super image created with shape: {img_set.shape}, dtype: {img_set.dtype}")
                    try:
                        im = Image.fromarray(img_set)
                        print(f"PIL Image created with size: {im.size}, mode: {im.mode}")
                        
                        fullpath = '%s/G_%s_%d_%d.png' % (self.image_dir, name, gen_iterations, i)
                        print(f"Saving image to: {fullpath}")
                        im.save(fullpath)
                        print(f"Successfully saved image to: {fullpath}")
                    except Exception as e:
                        print(f"Error creating/saving image: {e}")
                else:
                    print("img_set is None, skipping image creation")

            # Process discriminator view
            print("\nProcessing discriminator view...")
            i = -1  # Use highest resolution
            img = fake_imgs[i].detach()
            print(f"Using image at index {i} with shape: {img.shape}")
            
            print("Running image encoder...")
            region_features, _ = image_encoder(img)
            print(f"Region features shape: {region_features.shape}")
            
            att_sze = region_features.size(2)
            print(f"Attention size: {att_sze}")
            
            print("Computing words loss for attention maps...")
            _, _, att_maps = words_loss(region_features.detach(),
                                        words_embs.detach(),
                                        None, cap_lens,
                                        None, self.batch_size)
            print(f"Generated {len(att_maps)} attention maps")
            
            print("Building super images for discriminator view...")
            img_set, _ = \
                build_super_images(fake_imgs[i].detach().cpu(),
                                   captions, self.ixtoword, att_maps, att_sze)
                                   
            if img_set is not None:
                print(f"Super image created with shape: {img_set.shape}, dtype: {img_set.dtype}")
                try:
                    im = Image.fromarray(img_set)
                    print(f"PIL Image created with size: {im.size}, mode: {im.mode}")
                    
                    fullpath = '%s/D_%s_%d.png' % (self.image_dir, name, gen_iterations)
                    print(f"Saving image to: {fullpath}")
                    im.save(fullpath)
                    print(f"Successfully saved image to: {fullpath}")
                except Exception as e:
                    print(f"Error creating/saving discriminator image: {e}")
            else:
                print("img_set is None for discriminator view, skipping image creation")
                
        except Exception as e:
            print(f"Error in save_img_results: {e}")
            import traceback
            traceback.print_exc()

    def train(self):
        # 1. Build models & initialize training components
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        
        # Ensure encoders and models are in the correct mode
        text_encoder.train()
        image_encoder.train()
        
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        # Get hyperparameters from config
        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        lambda_supcon = cfg.TRAIN.SUPCON.LAMBDA  # Weight for supervised contrastive loss
        tau = cfg.TRAIN.SUPCON.TAU               # Temperature parameter
        
        # For mate sampler, actual batch size will be double the config batch size 
        # since each batch contains N anchors + N mates
        full_batch_size = batch_size * 2
        
        # Initialize noise vectors for the full batch size
        noise = torch.FloatTensor(full_batch_size, nz).to(self.device)
        fixed_noise = torch.FloatTensor(full_batch_size, nz).normal_(0, 1).to(self.device)

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            
            # Initialize counters for logging
            errD_total_sum = 0.0
            errG_total_sum = 0.0
            sup_loss_sum = 0.0
            
            data_iter = iter(self.data_loader)
            step = 0
            
            while step < self.num_batches:
                # Reset requires_grad for all discriminators
                self.set_requires_grad_value(netsD, True)
                
                #######################################################
                # (1) Prepare training data - handles the case where prepare_contrastive_data sorts
                #     and potentially disrupts the anchor-mate pairing
                #######################################################
                try:
                    data = next(data_iter)  # Use next() function instead of .next() method
                except StopIteration:
                    data_iter = iter(self.data_loader)
                    data = next(data_iter)
                
                # Instead of using prepare_contrastive_data, we'll manually process the data
                # to ensure anchor-mate pairs stay aligned
                if isinstance(data, dict):                          
                    imgs = data['imgs']
                    captions = data['captions']
                    cap_lens = data['cap_lens']
                    class_ids = data['class_ids']
                    keys = data['keys']
                else:                                              
                    imgs, captions, cap_lens, class_ids, keys = data
                
                # Convert to tensors if needed
                if not isinstance(cap_lens, torch.Tensor):
                    cap_lens = torch.tensor(cap_lens)
                if not isinstance(class_ids, torch.Tensor):
                    class_ids = torch.tensor(class_ids)
                if not isinstance(captions, torch.Tensor):
                    captions = torch.tensor(captions)
                
                # Move to device
                captions = captions.to(self.device)
                cap_lens = cap_lens.to(self.device)
                class_ids = class_ids.to(self.device)
                
                # Ensure captions are 2D [batch_size, sequence_length] as expected by LSTM
                if captions.dim() > 2:
                    print(f"Reshaping captions from {captions.shape} to 2D tensor")
                    captions = captions.squeeze(-1)  # Remove last dimension if it's 1
                
                # Check batch size and ensure it's even
                B = captions.size(0)
                if B % 2 != 0:
                    print(f"Warning: Batch size {B} is not even. Skipping this batch.")
                    continue
                    
                N = B // 2  # Half batch size - true split between anchors and mates
                
                # Process images - convert to torch tensor and move to device
                real_imgs = []
                for img_scale in imgs:
                    if not isinstance(img_scale, torch.Tensor):
                        img_scale = torch.tensor(img_scale)
                    img_scale = img_scale.to(self.device)
                    real_imgs.append(img_scale)
                
                # Split the batch into two halves (anchors and mates)
                # Note: We're assuming the MateBatchSampler creates batches with all anchors followed by all mates
                imgs_a = [scale[:N] for scale in real_imgs]
                imgs_m = [scale[N:] for scale in real_imgs]
                caps_a = captions[:N]  # First half of batch
                caps_m = captions[N:]  # Second half of batch
                lens_a = cap_lens[:N]
                lens_m = cap_lens[N:]
                cid_a = class_ids[:N]
                cid_m = class_ids[N:]
                
                # Verify tensor dimensions
                print(f"Anchor batch shapes: imgs_a[0]={imgs_a[0].shape}, caps_a={caps_a.shape}, lens_a={lens_a.shape}")
                print(f"Mate batch shapes: imgs_m[0]={imgs_m[0].shape}, caps_m={caps_m.shape}, lens_m={lens_m.shape}")
                
                #######################################################
                # (2) Compute text embeddings for both branches
                #######################################################
                # Sort each branch by caption length (required for proper LSTM packing)
                # Branch 1 (anchors)
                sorted_lens_a, sorted_idx_a = torch.sort(lens_a, dim=0, descending=True)
                sorted_caps_a = caps_a[sorted_idx_a]
                sorted_imgs_a = [img[sorted_idx_a] for img in imgs_a]
                sorted_cid_a = cid_a[sorted_idx_a]
                
                # Branch 2 (mates)
                sorted_lens_m, sorted_idx_m = torch.sort(lens_m, dim=0, descending=True)
                sorted_caps_m = caps_m[sorted_idx_m]
                sorted_imgs_m = [img[sorted_idx_m] for img in imgs_m]
                sorted_cid_m = cid_m[sorted_idx_m]
                
                # Process text embeddings for anchors
                hidden_a = text_encoder.init_hidden(N)
                words_embs_a, sent_emb_a = text_encoder(sorted_caps_a, sorted_lens_a, hidden_a)
                
                # Process text embeddings for mates
                hidden_m = text_encoder.init_hidden(N) 
                words_embs_m, sent_emb_m = text_encoder(sorted_caps_m, sorted_lens_m, hidden_m)
                
                # Create attention masks
                mask_a = (sorted_caps_a == 0)  # Shape is [batch_size, seq_len]
                num_words_a = words_embs_a.size(2)
                if mask_a.size(1) > num_words_a:
                    mask_a = mask_a[:, :num_words_a]
                elif mask_a.size(1) < num_words_a:
                    # Pad mask to match words_embs size
                    padding = torch.zeros(N, num_words_a - mask_a.size(1), dtype=torch.bool, device=self.device)
                    mask_a = torch.cat([mask_a, padding], dim=1)
                
                mask_m = (sorted_caps_m == 0)  # Shape is [batch_size, seq_len]
                num_words_m = words_embs_m.size(2)
                if mask_m.size(1) > num_words_m:
                    mask_m = mask_m[:, :num_words_m]
                elif mask_m.size(1) < num_words_m:
                    # Pad mask to match words_embs size
                    padding = torch.zeros(N, num_words_m - mask_m.size(1), dtype=torch.bool, device=self.device)
                    mask_m = torch.cat([mask_m, padding], dim=1)
                
                # Align sequence lengths for both branches
                seq_len_a = words_embs_a.size(2)
                seq_len_m = words_embs_m.size(2)
                print(f"Sequence lengths - anchor: {seq_len_a}, mate: {seq_len_m}")
                
                # Use the minimum length for both branches
                common_seq_len = min(seq_len_a, seq_len_m)
                
                if seq_len_a > common_seq_len:
                    words_embs_a = words_embs_a[:, :, :common_seq_len]
                    if mask_a.size(1) > common_seq_len:
                        mask_a = mask_a[:, :common_seq_len]
                
                if seq_len_m > common_seq_len:
                    words_embs_m = words_embs_m[:, :, :common_seq_len]
                    if mask_m.size(1) > common_seq_len:
                        mask_m = mask_m[:, :common_seq_len]
                
                #######################################################
                # (3) Generate fake images for both branches
                #######################################################
                # Update noise for both branches
                noise.normal_()
                
                # Instead of processing two separate branches which causes issues, we'll process 
                # all samples together in a single batch
                
                # Combine sorted captions and caption lengths
                all_caps = torch.cat([sorted_caps_a, sorted_caps_m], dim=0)
                all_lens = torch.cat([sorted_lens_a, sorted_lens_m], dim=0)
                all_cids = torch.cat([sorted_cid_a, sorted_cid_m], dim=0)
                
                # Resort in descending order of length (required for LSTM packing)
                # This is a second sorting on top of the separate anchor/mate sorts
                sorted_lens, sorted_idx = torch.sort(all_lens, dim=0, descending=True)
                all_caps = all_caps[sorted_idx]
                all_cids = all_cids[sorted_idx]
                
                # Get embeddings for all samples together
                print(f"Processing all {B} samples in a single batch")
                hidden = text_encoder.init_hidden(B)
                words_embs, sent_emb = text_encoder(all_caps, sorted_lens, hidden)
                
                # Create attention mask
                mask = (all_caps == 0)  # Shape is [batch_size, seq_len]
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                    
                print(f"Full batch - words_embs: {words_embs.shape}, sent_emb: {sent_emb.shape}, mask: {mask.shape}")
                
                # Process all images together
                all_imgs = []
                for i in range(len(sorted_imgs_a)):
                    # Combine real images in the same order as the sorted captions
                    unsorted_imgs = torch.cat([sorted_imgs_a[i], sorted_imgs_m[i]], dim=0)
                    # Apply the secondary sorting
                    all_imgs.append(unsorted_imgs[sorted_idx])
                
                # Generate noise for the full batch
                noise.normal_()
                
                try:
                    # Generate all fake images in a single pass
                    print("Generating all fake images in a single batch...")
                    fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)
                    
                    print(f"Successfully generated all fake images: {len(fake_imgs)} scales")
                    for i, img in enumerate(fake_imgs):
                        print(f"fake_imgs[{i}] shape: {img.shape}")
                        
                except Exception as e:
                    print(f"Error in image generation: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                #######################################################
                # (4) Update D networks (using full batch approach)
                #######################################################
                errD_total = 0
                
                # Update discriminators with the full batch
                for i, netD in enumerate(netsD):
                    netD.zero_grad()
                    try:
                        # Use the combined images and fake images
                        errD = discriminator_loss(netD, all_imgs[i], fake_imgs[i],
                                               sent_emb, real_labels, fake_labels)
                        errD.backward()
                        optimizersD[i].step()
                        errD_total += errD.item()
                        print(f"Successfully updated discriminator {i} with full batch")
                    except Exception as e:
                        print(f"Error in discriminator {i}: {e}")
                        import traceback
                        traceback.print_exc()
                
                #######################################################
                # (5) Update G network with GAN and contrastive losses
                #######################################################
                # Stop discriminators from calculating gradients
                self.set_requires_grad_value(netsD, False)
                
                # Zero generator gradients
                netG.zero_grad()
                
                # Compute generator loss for the full batch
                try:
                    print("Computing generator loss for full batch...")
                    errG_total, G_logs = generator_loss(
                        netsD, image_encoder, fake_imgs, real_labels,
                        words_embs, sent_emb, match_labels, sorted_lens, all_cids
                    )
                    print(f"Generator loss: {errG_total.item()}")
                except Exception as e:
                    print(f"Error in generator loss: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Add KL divergence loss
                kl_loss = KL_loss(mu, logvar)
                print(f"KL loss: {kl_loss.item()}")
                errG_total += kl_loss
                
                # Extract image features for contrastive loss
                try:
                    print("Computing features for contrastive loss...")
                    with torch.no_grad():
                        _, img_code = image_encoder(fake_imgs[-1])
                    
                    print(f"Extracted image features: {img_code.shape}")
                    
                    # Normalize features for contrastive loss
                    img_code_norm = F.normalize(img_code, dim=1)
                    sent_emb_norm = F.normalize(sent_emb, dim=1)
                    
                    # Use normalized features for contrastive loss
                    Z_img = img_code_norm
                    Z_txt = sent_emb_norm
                    
                    print(f"Normalized features - Z_img: {Z_img.shape}, Z_txt: {Z_txt.shape}, all_cids: {all_cids.shape}")
                    
                    # Create positive mask based on class IDs
                    pos_mask = build_pos_mask(all_cids)
                    
                    # Compute supervised contrastive losses
                    sup_img_loss = sup_con_loss(Z_img, pos_mask, tau)
                    sup_txt_loss = sup_con_loss(Z_txt, pos_mask, tau)
                    sup_loss = sup_img_loss + sup_txt_loss
                    
                    print(f"Contrastive losses - image: {sup_img_loss.item()}, text: {sup_txt_loss.item()}")
                    
                    # Add contrastive loss to total generator loss
                    errG_total += lambda_supcon * sup_loss
                except Exception as e:
                    print(f"Error in contrastive loss calculation: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Backward pass and update generator
                errG_total.backward()
                optimizerG.step()
                
                # Update moving averages of generator parameters
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(p.data, alpha=0.001)
                
                # Increment counters and log totals
                errD_total_sum += errD_total
                errG_total_sum += errG_total.item()
                sup_loss_sum += sup_loss.item()
                
                # Log progress
                if gen_iterations % 100 == 0:
                    print(f'Epoch [{epoch}/{self.max_epoch}] Step [{step}/{self.num_batches}] '
                          f'D_loss: {errD_total:.4f} G_loss: {errG_total.item():.4f} '
                          f'SupCon: {sup_loss.item():.4f}')
                
                # Save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    
                    # Save images from anchor branch
                    self.save_img_results(netG, fixed_noise[:N], sent_emb_a,
                                          words_embs_a, mask_a, image_encoder,
                                          caps_a, lens_a, epoch, name='average_anchors')
                    
                    # Save images from mate branch
                    self.save_img_results(netG, fixed_noise[N:], sent_emb_m,
                                          words_embs_m, mask_m, image_encoder,
                                          caps_m, lens_m, epoch, name='average_mates')
                    
                    load_params(netG, backup_para)
                
                step += 1
                gen_iterations += 1
            
            # End of epoch
            end_t = time.time()
            
            # Print epoch summary
            print(f'[{epoch}/{self.max_epoch}][{self.num_batches}] '
                  f'Loss_D: {errD_total_sum/self.num_batches:.4f} '
                  f'Loss_G: {errG_total_sum/self.num_batches:.4f} '
                  f'SupCon: {sup_loss_sum/self.num_batches:.4f} '
                  f'Time: {end_t - start_t:.2f}s')
            
            # Save models periodically
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                self.save_model(netG, avg_param_G, netsD, epoch)
        
        # End of training - save final results and model
        print('Training completed. Saving final model...')
        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            try:
                s_tmp = '%s/single_samples/%s/%s' %\
                    (save_dir, split_dir, filenames[i])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)

                fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
                # range from [-1, 1] to [0, 1]
                # img = (images[i] + 1.0) / 2
                img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
                # range from [0, 1] to [0, 255]
                ndarr = img.permute(1, 2, 0).data.cpu().numpy()
                
                # Check if the array has valid shape and values
                if ndarr.ndim != 3 or ndarr.shape[2] not in [1, 3, 4]:
                    print(f"Warning: Invalid image shape {ndarr.shape} for {filenames[i]}")
                    continue
                    
                im = Image.fromarray(ndarr)
                im.save(fullpath)
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                # Continue with the next image instead of crashing

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '' or not os.path.exists(cfg.TRAIN.NET_G):
            print(f'Error: the generator model path is not found! Path: {cfg.TRAIN.NET_G}')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG = netG.to(self.device)
            netG.eval()
            #
            # First load the pre-trained text encoder state dict to get its vocabulary size
            state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            # Get the vocabulary size from the pre-trained model
            encoder_weight_shape = state_dict['encoder.weight'].shape
            vocab_size = encoder_weight_shape[0]
            print(f"Pre-trained model vocabulary size: {vocab_size}")
            
            # Initialize text encoder with the vocabulary size from the pre-trained model
            text_encoder = RNN_ENCODER(vocab_size, nhidden=cfg.TEXT.EMBEDDING_DIM)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to(self.device)
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.to(self.device)

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            # Use a simpler directory structure
            save_dir = 'sampling'
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break

                    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        # Extract bird class from the key
                        bird_class = keys[j].split('/')[0] if '/' in keys[j] else keys[j]
                        # Create a simpler filename
                        filename = f"{bird_class}_{j}.png"
                        
                        # Process the image
                        k = -1  # Use the highest resolution
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        
                        # Save to the sampling directory
                        fullpath = os.path.join(save_dir, filename)
                        im.save(fullpath)
                        print(f"Saved image to: {fullpath}")

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.to(self.device)
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG = netG.to(self.device)
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                # Extract data from the dictionary
                # Handle different types of sorted_indices (could be a tensor, numpy array, or list)
                captions, cap_lens, sorted_indices = data_dic[key]
                
                # Convert sorted_indices to a numpy array if it's not already
                if isinstance(sorted_indices, torch.Tensor):
                    sorted_indices = sorted_indices.cpu().numpy()
                elif isinstance(sorted_indices, list):
                    sorted_indices = np.array(sorted_indices)

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.to(self.device)
                cap_lens = cap_lens.to(self.device)
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.to(self.device)
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)