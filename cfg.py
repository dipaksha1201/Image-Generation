from easydict import EasyDict as edict

# Create a complete configuration for the image generation project
cfg = edict()

# Basic configuration
cfg.CONFIG_NAME = 'image_gen'
cfg.DATASET_NAME = 'birds'
cfg.DATA_DIR = 'data'
cfg.CUDA = True
cfg.GPU_ID = 0

# RNN config
cfg.RNN_TYPE = 'LSTM'  # Options: 'LSTM', 'GRU'
cfg.B_VALIDATION = False

# Training config
cfg.TRAIN = edict()
cfg.TRAIN.FLAG = True  # Important: set to True to enable model_dir and image_dir creation
cfg.TRAIN.NET_G = 'Model/bird_AttnGAN2.pth'  # Path to generator model
cfg.TRAIN.NET_E = 'DAMSMencoders/bird/text_encoder200.pth'  # Path to text encoder
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.MAX_EPOCH = 2  # Set to 1 epochs as in the YAML file
cfg.TRAIN.SNAPSHOT_INTERVAL = 50
cfg.TRAIN.DISCRIMINATOR_LR = 0.0002
cfg.TRAIN.GENERATOR_LR = 0.0002
cfg.TRAIN.B_NET_D = True

# Supervised contrastive learning config
cfg.TRAIN.SUPCON = edict()
cfg.TRAIN.SUPCON.LAMBDA = 1
cfg.TRAIN.SUPCON.TAU = 0.7

# Loss function smoothing parameters
cfg.TRAIN.SMOOTH = edict()
cfg.TRAIN.SMOOTH.GAMMA1 = 5.0  # Attention smoothing factor
cfg.TRAIN.SMOOTH.GAMMA2 = 5.0  # Similarity smoothing factor
cfg.TRAIN.SMOOTH.GAMMA3 = 10.0  # For matching loss
cfg.TRAIN.SMOOTH.LAMBDA = 1.0  # Weight for matching loss

# KL divergence coefficient
cfg.TRAIN.COEFF = edict()
cfg.TRAIN.COEFF.KL = 2.0  # From YAML file

# Data loading workers
cfg.WORKERS = 4

# GAN configuration
cfg.GAN = edict()
cfg.GAN.Z_DIM = 100
cfg.GAN.CONDITION_DIM = 100
cfg.GAN.B_ATTENTION = True
cfg.GAN.B_DCGAN = False
cfg.GAN.DF_DIM = 16  # Reduced from 64 to match pre-trained model
cfg.GAN.GF_DIM = 32  # Reduced from 128 to match pre-trained model
cfg.GAN.R_NUM = 2  # Number of residual blocks

# Text configuration
cfg.TEXT = edict()
cfg.TEXT.EMBEDDING_DIM = 256
cfg.TEXT.WORDS_NUM = 18  # Maximum number of words in a caption
cfg.TEXT.CAPTIONS_PER_IMAGE = 10

# Tree configuration for multi-stage generation
cfg.TREE = edict()
cfg.TREE.BRANCH_NUM = 3
cfg.TREE.BASE_SIZE = 64

# Stage-specific configurations
cfg.STAGE1 = edict()
cfg.STAGE1.G_GF_DIM = 32  # Reduced from 128 to match pre-trained model
cfg.STAGE1.G_DF_DIM = 16  # Reduced from 64 to match pre-trained model
cfg.STAGE1.D_DF_DIM = 16  # Added from YAML (reduced from 64)
cfg.STAGE1.D_GF_DIM = 32  # Added from YAML (reduced from 128)
cfg.STAGE1.RATIO = 0.5   # Added from YAML

cfg.STAGE2 = edict()
cfg.STAGE2.G_GF_DIM = 32  # Reduced from 128 to match pre-trained model
cfg.STAGE2.G_DF_DIM = 16  # Reduced from 64 to match pre-trained model
cfg.STAGE2.D_DF_DIM = 16  # Added from YAML (reduced from 64)
cfg.STAGE2.D_GF_DIM = 32  # Added from YAML (reduced from 128)
cfg.STAGE2.RATIO = 0.5   # Added from YAML

cfg.STAGE3 = edict()
cfg.STAGE3.G_GF_DIM = 32  # Reduced from 128 to match pre-trained model
cfg.STAGE3.G_DF_DIM = 16  # Reduced from 64 to match pre-trained model
cfg.STAGE3.D_DF_DIM = 16  # Added from YAML (reduced from 64)
cfg.STAGE3.D_GF_DIM = 32  # Added from YAML (reduced from 128)
cfg.STAGE3.RATIO = 0.5   # Added from YAML
