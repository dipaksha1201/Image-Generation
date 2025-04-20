from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from cfg import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    try:
        # Print data types for debugging
        print(f"\nPreparing data:")
        if isinstance(data, dict):
            imgs = data.get('imgs', [])
            captions = data.get('captions', None)
            captions_lens = data.get('cap_lens', None)
            class_ids = data.get('class_ids', None)
            keys = data.get('keys', None)
            print(f"Data is a dictionary with keys: {list(data.keys())}")
        else:
            imgs, captions, captions_lens, class_ids, keys = data
        
        # Ensure captions_lens is a tensor
        if not isinstance(captions_lens, torch.Tensor):
            print(f"Warning: captions_lens is not a tensor but {type(captions_lens)}")
            if isinstance(captions_lens, (list, np.ndarray)):
                captions_lens = torch.tensor(captions_lens)
            else:
                print(f"Converting {captions_lens} to tensor")
                captions_lens = torch.tensor([captions_lens])
        
        # Sort data by the length in a decreasing order
        print(f"Sorting with captions_lens of type {type(captions_lens)}")
        sorted_cap_lens, sorted_cap_indices = \
            torch.sort(captions_lens, dim=0, descending=True)

        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            # Check if CUDA or MPS is available
            if hasattr(torch, 'mps') and torch.backends.mps.is_available() and cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).to('mps'))
            elif torch.cuda.is_available() and cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).cuda())
            else:
                real_imgs.append(Variable(imgs[i]))

        captions = captions[sorted_cap_indices].squeeze()
        class_ids = class_ids[sorted_cap_indices].numpy()
        # sent_indices = sent_indices[sorted_cap_indices]
        keys = [keys[i] for i in sorted_cap_indices.numpy()]
        
        # Check if CUDA or MPS is available
        if hasattr(torch, 'mps') and torch.backends.mps.is_available() and cfg.CUDA:
            captions = Variable(captions).to('mps')
            sorted_cap_lens = Variable(sorted_cap_lens).to('mps')
        elif torch.cuda.is_available() and cfg.CUDA:
            captions = Variable(captions).cuda()
            sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        else:
            captions = Variable(captions)
            sorted_cap_lens = Variable(sorted_cap_lens)

        return [real_imgs, captions, sorted_cap_lens,
                class_ids, keys]
    except Exception as e:
        print(f"Error in prepare_data: {e}")
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"{k}: {type(v)}")
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                print(f"Item {i}: {type(item)}")
        # Return a simple placeholder
        print("Returning placeholder data to prevent crash")
        device = 'cuda' if torch.cuda.is_available() and cfg.CUDA else 'mps' if hasattr(torch, 'mps') and torch.backends.mps.is_available() and cfg.CUDA else 'cpu'
        return [
            [Variable(torch.zeros(1, 3, 64, 64)).to(device)],  # real_imgs
            Variable(torch.zeros(1, 18)).long().to(device),  # captions
            Variable(torch.tensor([10])).to(device),  # sorted_cap_lens
            np.array([0]),  # class_ids
            ["placeholder"]  # keys
        ]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        # Make sure we don't try to access indices beyond what's available
        branch_num = min(cfg.TREE.BRANCH_NUM, len(imsize))
        for i in range(branch_num):
            # print(imsize[i])
            if i < (branch_num - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        # In Python 3, strings are already Unicode
                        # Just filter out non-ASCII characters
                        t = ''.join([c for c in t if ord(c) < 128])
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=pickle.HIGHEST_PROTOCOL)
                print('Save to: ', filepath)
        else:
            try:
                with open(filepath, 'rb') as f:
                    x = pickle.load(f)
                    train_captions, test_captions = x[0], x[1]
                    ixtoword, wordtoix = x[2], x[3]
                    del x
                    n_words = len(ixtoword)
                    print('Load from: ', filepath)
            except UnicodeDecodeError:
                # Try to load Python 2 pickle in Python 3
                with open(filepath, 'rb') as f:
                    x = pickle.load(f, encoding='latin1')
                    train_captions, test_captions = x[0], x[1]
                    ixtoword, wordtoix = x[2], x[3]
                    del x
                    n_words = len(ixtoword)
                    print('Load from (Python 2 pickle): ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        print(f"Loading class IDs for {total_num} files...")
        if os.path.isfile(data_dir + '/class_info.pickle'):
            try:
                print(f"Attempting to load class_info.pickle from {data_dir}")
                with open(data_dir + '/class_info.pickle', 'rb') as f:
                    class_id = pickle.load(f)
                print(f"Loaded class_id with length {len(class_id)}")
                
                # Ensure class_id has the right length
                if len(class_id) != total_num:
                    print(f"Warning: class_id length ({len(class_id)}) does not match filenames length ({total_num})")
                    print(f"Generating new class_id array with correct length")
                    class_id = np.arange(total_num)  # Fallback to default
            except Exception as e:
                print(f"Error loading class_info.pickle: {e}")
                print(f"Generating default class_id array")
                class_id = np.arange(total_num)  # Fallback to default
        else:
            print(f"No class_info.pickle found, generating default class_id array")
            class_id = np.arange(total_num)
            
        print(f"Final class_id array length: {len(class_id)}")
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        print(f"Looking for filenames.pickle at: {filepath}")
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'rb') as f:
                    filenames = pickle.load(f)
                print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
            except Exception as e:
                print(f"Error loading filenames from {filepath}: {e}")
                filenames = []
        else:
            print(f"File not found: {filepath}")
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        # Handle potential mismatch between filenames and bounding boxes
        if self.bbox is not None:
            try:
                bbox = self.bbox[key]
                data_dir = '%s/CUB_200_2011' % self.data_dir
            except KeyError:
                # If the key is not found in the bbox dictionary, use None
                print(f"Warning: No bounding box found for {key}, using None instead")
                bbox = None
                data_dir = self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        # Always look in the CUB_200_2011/images directory
        img_name = '%s/CUB_200_2011/images/%s.jpg' % (self.data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        
        # Convert cls_id to a tensor to ensure compatibility with PyTorch operations
        if isinstance(cls_id, (int, float, np.integer, np.floating)):
            # For single values, create a scalar tensor
            cls_id_tensor = torch.tensor(cls_id, dtype=torch.long)
        elif isinstance(cls_id, np.ndarray):
            # Ensure no extra dimensions are added
            cls_id_tensor = torch.from_numpy(cls_id).long().squeeze()
        else:
            # For other types (lists, etc.)
            cls_id_tensor = torch.tensor(cls_id, dtype=torch.long).squeeze()
        
        # Return a dictionary instead of a tuple to match what main.py expects
        return {
            'imgs': imgs,
            'captions': caps,
            'cap_lens': cap_len,
            'class_ids': cls_id_tensor,
            'keys': key
        }

    def __len__(self):
        return len(self.filenames)
