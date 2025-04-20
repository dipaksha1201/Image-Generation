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

from losses import words_loss
from losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys

# ################# Text to image task############################ #
class condGANTrainer(object):
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
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        # Move labels to the appropriate device
        real_labels = real_labels.to(self.device)
        fake_labels = fake_labels.to(self.device)
        match_labels = match_labels.to(self.device)

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
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        # Move tensors to the appropriate device
        noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device)

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                try:
                    print("Fetching next batch...")
                    data = next(data_iter)  # Use next() function instead of .next() method
                except StopIteration:
                    print("Restarting data iterator...")
                    data_iter = iter(self.data_loader)
                    data = next(data_iter)
                print("Successfully retrieved batch data")
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                print("Running text encoder...")
                try:
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    print("Text encoding complete")
                except Exception as e:
                    print(f"Error in text encoder: {e}")
                    raise
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                print("Generating noise...")
                noise.data.normal_(0, 1)
                print("Running generator model...")
                try:
                    fake_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)
                    print("Generator model completed successfully")
                except Exception as e:
                    print(f"Error in generator model: {e}")
                    raise

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                print("Updating discriminator networks...")
                for i in range(len(netsD)):
                    print(f"Processing discriminator {i}...")
                    netsD[i].zero_grad()
                    try:
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  sent_emb, real_labels, fake_labels)
                        print(f"Discriminator {i} loss calculated")
                        # backward
                        errD.backward()
                        optimizersD[i].step()
                        errD_total += errD
                        print(f"errD is of type {type(errD)} with shape {errD.shape if hasattr(errD, 'shape') else 'no shape'}")
                        D_logs += 'errD%d: %.2f ' % (i, errD.item())
                    except Exception as e:
                        print(f"Error in discriminator {i}: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                print("Computing generator loss...")
                netG.zero_grad()
                try:
                    errG_total, G_logs = \
                        generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                      words_embs, sent_emb, match_labels, cap_lens, class_ids)
                    print(f"Generator loss computed: {errG_total.item()}")
                except Exception as e:
                    print(f"Error in generator_loss: {e}")
                    raise
                # Skip the redundant words_loss call
                # The words_loss is already calculated in generator_loss
                print("Skipping redundant words_loss calculation - already done in generator_loss")
                # Use the losses from generator_loss instead
                # We don't need attn_maps here since they're not used
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                print("Updating moving average parameters...")
                try:
                    for p, avg_p in zip(netG.parameters(), avg_param_G):
                        # Using newer PyTorch syntax for in-place add with scaling
                        avg_p.mul_(0.999).add_(p.data, alpha=0.001)
                    print("Moving average parameters updated successfully")
                except Exception as e:
                    print(f"Error updating parameters: {e}")
                    raise

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens,
                                          epoch, name='current')
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens,
                                          epoch, name='current')
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