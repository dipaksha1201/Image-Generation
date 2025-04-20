import torch
from torch.utils.data import Sampler
import random
import numpy as np
from collections import defaultdict
from cfg import cfg

def prepare_contrastive_data(data):
    """
    Standardises a mini‑batch coming from TextDataset + MateBatchSampler.

    Returns
    -------
    real_imgs  : list[Tensor]  –  one 4‑D tensor per resolution, on CUDA/MPS/CPU
    captions   : LongTensor    –  shape [B, max_len]  (same device as images)
    cap_lens   : LongTensor    –  shape [B]           (same device)
    class_ids  : LongTensor    –  shape [B]           (same device)
    keys       : list[str]     –  image keys in sorted order
    """
    # --------------------------------------------------------- unpack
    if isinstance(data, dict):                           # collate_fn dict mode
        imgs       = data['imgs']
        captions   = data['captions']
        cap_lens   = data['cap_lens']
        class_ids  = data['class_ids']
        keys       = data['keys']
    else:                                                # legacy tuple mode
        imgs, captions, cap_lens, class_ids, keys = data

    # ensure tensors
    if not isinstance(cap_lens, torch.Tensor):
        cap_lens = torch.as_tensor(cap_lens)
    if not isinstance(class_ids, torch.Tensor):
        class_ids = torch.as_tensor(class_ids)
    
    # Convert to device (CPU, CUDA, or MPS)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # --------------------------------------------------------- sort by length
    # Sort in descending order (required for pack_padded_sequence)
    sorted_lens, sorted_idx = torch.sort(cap_lens, dim=0, descending=True)

    # Process images
    real_imgs = []
    for img_scale in imgs:
        if isinstance(img_scale, torch.Tensor):
            img_scale = img_scale[sorted_idx].to(device)
            real_imgs.append(img_scale)
        else:
            # Handle case where imgs might be a list of numpy arrays
            img_tensor = torch.tensor(img_scale)[sorted_idx].to(device)
            real_imgs.append(img_tensor)

    # Process text data
    if isinstance(captions, torch.Tensor):
        captions = captions[sorted_idx].to(device)
    else:
        captions = torch.tensor(captions, dtype=torch.long)[sorted_idx].to(device)
        
    # Ensure captions are 2D [batch_size, sequence_length] as expected by LSTM
    if captions.dim() > 2:
        print(f"Reshaping captions from {captions.shape} to 2D tensor")
        # If it's [batch, seq_len, 1], reshape to [batch, seq_len]
        captions = captions.squeeze(-1)
    elif captions.dim() == 1:
        # If it's [batch], reshape to [batch, 1]
        captions = captions.unsqueeze(1)
        
    cap_lens = sorted_lens.to(device)
    class_ids = class_ids[sorted_idx].to(device)

    # Process keys
    if isinstance(sorted_idx, torch.Tensor):
        sorted_idx = sorted_idx.cpu().numpy()
    keys = [keys[i] for i in sorted_idx]

    return real_imgs, captions, cap_lens, class_ids, keys

def build_pos_mask(class_ids: torch.Tensor) -> torch.BoolTensor:
    """
    Single‑label supervised‑contrastive positive mask.

    Parameters
    ----------
    class_ids : 1‑D LongTensor, shape [B]
        Semantic label (one integer) for each sample in the batch.

    Returns
    -------
    pos_mask : BoolTensor, shape [B, B]
        pos_mask[i, j] = True  ⇔  i ≠ j and class_ids[i] == class_ids[j]
        (diagonal is always False).
    """
    if class_ids.dim() != 1:
        raise ValueError("class_ids must be 1‑D for single‑label mode.")

    # Broadcast compare: [1, B] == [B, 1] → [B, B]
    cid   = class_ids.view(1, -1)
    mask  = cid == cid.T          # BoolTensor, same device as class_ids
    mask.fill_diagonal_(False)    # remove self‑matches
    return mask

class MateBatchSampler(Sampler):
    """Yield batches that contain *N anchors* + *N mates* (same class).

    Parameters
    ----------
    dataset : any object with a ``class_id`` attribute indexable by dataset
              index (e.g., your ``TextDataset``).
    batch_size : int
        Number of **anchors**.  Returned batch length is ``2*batch_size``.
    shuffle : bool, default True
        Randomise anchor order each epoch.
    drop_last : bool, default True
        Whether to drop the last incomplete anchor batch.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        self.dataset      = dataset
        self.anchor_bs    = batch_size
        self.shuffle      = shuffle
        self.drop_last    = drop_last

        # build label -> list[index] lookup once
        self.label_to_idx = defaultdict(list)
        
        # Create an index -> class_id mapping to use in __iter__
        self.idx_to_class = {}
        
        # Check if 'class_id' is an attribute or directly accessible
        if hasattr(dataset, 'class_id'):
            # Direct access
            print(f"Using dataset.class_id directly (length {len(dataset.class_id)})")
            for idx, cid in enumerate(dataset.class_id):
                cid_int = int(cid)
                self.label_to_idx[cid_int].append(idx)
                self.idx_to_class[idx] = cid_int
        else:
            # Try alternatives or sample from the dataset
            print("Warning: dataset doesn't have class_id attribute, trying alternatives")
            # Try to get class IDs by sampling from dataset
            for idx in range(len(dataset)):
                # Get sample and extract class_id - handle both dict and tuple return types
                sample = dataset[idx]
                if isinstance(sample, dict):
                    cid = sample['class_ids']
                elif isinstance(sample, (tuple, list)) and len(sample) >= 4:
                    cid = sample[3]  # Assuming 4th element is class_id
                else:
                    raise ValueError(f"Cannot extract class_id from dataset samples, got {type(sample)}")
                
                # Handle different types of class_id
                if isinstance(cid, torch.Tensor):
                    cid = cid.item() if cid.numel() == 1 else int(cid[0])
                elif isinstance(cid, np.ndarray):
                    cid = int(cid[0]) if cid.size > 0 else 0
                
                cid_int = int(cid)    
                self.label_to_idx[cid_int].append(idx)
                self.idx_to_class[idx] = cid_int
                
            print(f"Built label_to_idx map with {len(self.label_to_idx)} unique classes")

        # pre‑compute epoch order container
        self.idxs = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.idxs)

        # walk anchors in steps of anchor_bs
        for start in range(0, len(self.idxs), self.anchor_bs):
            anchor_idx = self.idxs[start:start + self.anchor_bs]
            if len(anchor_idx) < self.anchor_bs and self.drop_last:
                break  # skip final incomplete batch

            # pick one same‑class mate for each anchor
            try:
                mate_idx = [
                    # Use our idx_to_class mapping instead of accessing dataset.class_id directly
                    random.choice(self.label_to_idx[self.idx_to_class[i]])
                    for i in anchor_idx
                ]
                yield anchor_idx + mate_idx   # length 2*anchor_bs
            except Exception as e:
                print(f"Error in MateBatchSampler.__iter__: {e}")
                print(f"anchor_idx: {anchor_idx}")
                # Skip this batch if there's an error
                continue

    def __len__(self):
        n_anchor_batches = len(self.dataset) // self.anchor_bs
        if not self.drop_last and len(self.dataset) % self.anchor_bs:
            n_anchor_batches += 1
        return n_anchor_batches  # Return the calculated value

def sup_con_loss(z: torch.Tensor,
                 pos_mask: torch.BoolTensor,
                 tau: float = 0.07,
                 eps:  float = 1e-8) -> torch.Tensor:
    """
    Supervised‑contrastive loss (single‑label version)

    Parameters
    ----------
    z         : Tensor, shape [B, D]
        Feature vectors for one view (image OR text OR mixed).
        They should already be L2‑normalised:  z = F.normalize(z, dim=1).
    pos_mask  : BoolTensor, shape [B, B]
        pos_mask[i,j] == True  ⇔  i ≠ j and samples i & j share the label.
        (Get it from build_pos_mask(class_ids_full).)
    tau       : float
        Temperature hyper‑parameter (defaults to 0.07 as in SimCLR/SupCon).
    eps       : float
        Small value to avoid divide‑by‑zero when an anchor has no positives.

    Returns
    -------
    loss : scalar Tensor
        The batch‑averaged supervised contrastive loss.
    """
    B = z.size(0)

    # ---------------------------------------------------------------------
    # 1. Pair‑wise cosine similarities, scaled by temperature
    #    sim[i,j] = z_i · z_j / tau
    # ---------------------------------------------------------------------
    sim = torch.mm(z, z.t()).clamp(min=-1.0, max=1.0) / tau          # [B, B]

    # 2. Mask out self‑similarities so they never enter the softmax.
    #    A simple way is to subtract a huge number on the diagonal.
    # ---------------------------------------------------------------------
    sim = sim - torch.eye(B, device=z.device) * 1e9                  # [B, B]

    # 3. Log‑softmax over every row  (log p_{i→j})
    # ---------------------------------------------------------------------
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)       # [B, B]

    # 4. Compute the mean log‑probability of positive targets
    # ---------------------------------------------------------------------
    pos_mask_f = pos_mask.float()                                    # [B, B]
    num_pos    = pos_mask_f.sum(1)                                   # [B]
    # avoid 0/0 when an anchor has no positives
    mean_log_prob_pos = (pos_mask_f * log_prob).sum(1) / (num_pos + eps)  # [B]

    # 5. Loss is the negative of that, averaged over valid anchors only
    # ---------------------------------------------------------------------
    loss = -mean_log_prob_pos[num_pos > 0].mean()

    return loss