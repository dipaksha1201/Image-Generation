from __future__ import print_function

from utils import mkdir_p, build_super_images, cfg_from_file
from losses import sent_loss, words_loss

from contrastive_encoders import prepare_contrastive_data
import torch.nn.functional as F
from model import RNN_ENCODER, CNN_ENCODER
from contrastive_encoders import build_pos_mask, sup_con_loss

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from main import load_data

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='encoder.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def train(
    dataloader,              # DataLoader with MateBatchSampler
    cnn_model,               # image encoder
    rnn_model,               # text encoder
    optimizer,               # torch.optim.Optimizer
    epoch: int,
    ixtoword: dict[int, str],
    image_dir: str
):
    """One full epoch of supervised‑contrastive DAMSM pre‑training."""

    cnn_model.train()
    rnn_model.train()

    # Running sums for logging
    s_tot0 = s_tot1 = w_tot0 = w_tot1 = sup_tot = 0.0
    start_time     = time.time()

    for step, data in enumerate(dataloader):
        try:
            # ---------------------------------------------------------------
            # (1) unpack & split anchor vs. mate
            # ---------------------------------------------------------------
            imgs, captions, cap_lens, class_ids, keys = prepare_contrastive_data(data)
            
            # Check batch size and make sure it's even
            B = class_ids.size(0)              # 2 * anchor_bs
            if B % 2 != 0:
                print(f"Warning: Batch size {B} is not even. Skipping this batch.")
                continue
                
            N = B // 2

            # Split the batch into two halves (anchors and mates)
            imgs_a  = [ scale[:N] for scale in imgs ]
            imgs_m  = [ scale[N:] for scale in imgs ]
            caps_a  = captions[:N]  # First half of batch
            caps_m  = captions[N:]  # Second half of batch
            lens_a  = cap_lens[:N]
            lens_m  = cap_lens[N:]
            cid_a   = class_ids[:N]
            cid_m   = class_ids[N:]

            # zero grads
            optimizer.zero_grad()

            # ---------------------------------------------------------------
            # (2) forward pass – anchors
            # ---------------------------------------------------------------
            w_feat_a, sent_code_a = cnn_model(imgs_a[-1])
            hidden_a = rnn_model.init_hidden(N)
            
            # Debug info
            print(f"Anchor batch: caps_a shape={caps_a.shape}, lens_a shape={lens_a.shape}")
            print(f"Hidden shape: {[h.shape for h in hidden_a if isinstance(h, torch.Tensor)]}")
            
            w_emb_a, sent_emb_a = rnn_model(caps_a, lens_a, hidden_a)

            # forward pass – mates
            w_feat_m, sent_code_m = cnn_model(imgs_m[-1])
            hidden_m = rnn_model.init_hidden(N)
            w_emb_m, sent_emb_m = rnn_model(caps_m, lens_m, hidden_m)
            
        except Exception as e:
            print(f"Error in batch {step}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # ---------------------------------------------------------------
        # (3) DAMSM losses (per half)
        # ---------------------------------------------------------------
        index_labels = torch.arange(N, device=class_ids.device)

        w0a, w1a, attn_a = words_loss(w_feat_a, w_emb_a, index_labels,
                                      lens_a, cid_a, N)
        s0a, s1a = sent_loss(sent_code_a, sent_emb_a, index_labels,
                              cid_a, N)

        w0m, w1m, attn_m = words_loss(w_feat_m, w_emb_m, index_labels,
                                      lens_m, cid_m, N)
        s0m, s1m = sent_loss(sent_code_m, sent_emb_m, index_labels,
                              cid_m, N)

        damsm_loss = w0a + w1a + s0a + s1a + w0m + w1m + s0m + s1m

        # ---------------------------------------------------------------
        # (4) supervised‑contrastive loss
        # ---------------------------------------------------------------
        Z_img = F.normalize(torch.cat([sent_code_a, sent_code_m], 0), dim=1)
        Z_txt = F.normalize(torch.cat([sent_emb_a, sent_emb_m], 0), dim=1)

        cid_full = torch.cat([cid_a, cid_m], 0)  # shape [B]
        pos_mask = build_pos_mask(cid_full)      # BoolTensor [B, B]

        sup_img = sup_con_loss(Z_img, pos_mask, cfg.TRAIN.SUPCON.TAU)
        sup_txt = sup_con_loss(Z_txt, pos_mask, cfg.TRAIN.SUPCON.TAU)
        sup_loss = sup_img + sup_txt

        # ---------------------------------------------------------------
        # (5) total loss & backward
        # ---------------------------------------------------------------
        loss = damsm_loss + cfg.TRAIN.SUPCON.LAMBDA * sup_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        # ---------------------------------------------------------------
        # (6) accumulate for logs
        # ---------------------------------------------------------------
        s_tot0 += (s0a + s0m).item();  s_tot1 += (s1a + s1m).item()
        w_tot0 += (w0a + w0m).item();  w_tot1 += (w1a + w1m).item()
        sup_tot += sup_loss.item()

        # ---------------------------------------------------------------
        # (7) periodic print & visuals
        # ---------------------------------------------------------------
        if step % UPDATE_INTERVAL == 0 and step > 0:
            elapsed = time.time() - start_time
            denom   = UPDATE_INTERVAL
            print(
                f"| epoch {epoch:3d} | {step:5d}/{len(dataloader):5d} batches "
                f"| ms/batch {elapsed*1000/denom:5.1f} |" \
                f" s_loss {s_tot0/denom:5.2f} {s_tot1/denom:5.2f} |" \
                f" w_loss {w_tot0/denom:5.2f} {w_tot1/denom:5.2f} |" \
                f" sup {sup_tot/denom:5.2f}")

            # reset counters
            s_tot0 = s_tot1 = w_tot0 = w_tot1 = sup_tot = 0.0
            start_time = time.time()

            # attention maps (anchors only)
            nef, att_size = w_feat_a.size(1), w_feat_a.size(2)
            img_set, _ = build_super_images(imgs_a[-1].cpu(), caps_a,
                                             ixtoword, attn_a, att_size)
            if img_set is not None:
                os.makedirs(image_dir, exist_ok=True)
                Image.fromarray(img_set).save(os.path.join(
                    image_dir, f"attention_maps_epoch{epoch}_step{step}.png"))

    # end for step
    return (epoch + 1) * len(dataloader)

def evaluate(dataloader,
             cnn_model,
             rnn_model):
    """Return (sentence_loss, word_loss, sup_loss) averaged over loader."""
    cnn_model.eval()
    rnn_model.eval()

    w_sum = s_sum = sup_sum = 0.0
    n_steps = 0

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            real_imgs, captions, cap_lens, class_ids, _ = prepare_contrastive_data(data)
            B = class_ids.size(0)
            words_feat, sent_code = cnn_model(real_imgs[-1])

            hidden = rnn_model.init_hidden(B)
            words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

            # labels for DAMSM cross‑entropy (index‑of‑self)
            labels = torch.arange(B, device=captions.device)

            w0, w1, _ = words_loss(words_feat, words_emb, labels,
                                    cap_lens, class_ids, B)
            s0, s1 = sent_loss(sent_code, sent_emb, labels, class_ids, B)

            w_sum += (w0 + w1).item()
            s_sum += (s0 + s1).item()

            # optional: SupCon loss if batch came from MateBatchSampler (even)
            if B % 2 == 0:
                Z_img = F.normalize(sent_code, dim=1)
                pos_mask = build_pos_mask(class_ids)
                sup_sum += sup_con_loss(Z_img, pos_mask, cfg.TRAIN.SUPCON.TAU).item()
            n_steps += 1
            if step == 50:  # cap evaluation cost
                break

    return s_sum / n_steps, w_sum / n_steps, sup_sum / max(1, n_steps)

def build_models(device):
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = torch.arange(batch_size, dtype=torch.long)
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=device)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name, map_location=device)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    # Move models to the appropriate device
    text_encoder = text_encoder.to(device)
    image_encoder = image_encoder.to(device)
    labels = labels.to(device)

    return text_encoder, image_encoder, labels, start_epoch

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file:
        cfg = cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.CUDA = True
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    # Set seeds for reproducibility
    if torch.cuda.is_available() and cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'encoder_output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    # Device is defined later
    cudnn.benchmark = True

    # Data loader ##################################################
    dataloader, dataset = load_data(split='train', shuffle=True)
    dataloader_val, _ = load_data(split='test', shuffle=True)
    batch_size = cfg.TRAIN.BATCH_SIZE  # define batch_size for training and evaluation calls

    # Train ##############################################################
    # Define device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) device")
    elif torch.cuda.is_available() and cfg.CUDA:
        device = torch.device(f"cuda:{cfg.GPU_ID}")
        print(f"Using CUDA device {cfg.GPU_ID}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    text_encoder, image_encoder, labels, start_epoch = build_models(device)
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          optimizer, epoch,
                          dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss, sup_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, sup_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                (epoch + 1) == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch + 1))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch + 1))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
