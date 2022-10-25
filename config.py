# --------------------------------------------------------
# IFNet
# Written by Jiaheng Wang
# --------------------------------------------------------

import os
import random
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

#Path to dataset
_C.DATA.DATA_PATH = '~/Datasets/MI/BCIC/'
#_C.DATA.DATA_PATH = '~/Datasets/MI/openbmi/'
_C.DATA.TRAIN_FILES = ['training.mat']
_C.DATA.TEST_FILES = ['evaluation.mat']

_C.DATA.BATCH_SIZE = 32
_C.DATA.RTA = 5 #repeated trial augmentation
_C.DATA.K_FOLD = 5
_C.DATA.FOLD_STEP = 1
_C.DATA.BLOCK = True    #block-wise cv

_C.DATA.FILTER_BANK = [(4, 16), (16, 40)]
_C.DATA.FS = 250
_C.DATA.RESAMPLE = 1
_C.DATA.REF_DUR = 0 #time duration of baseline reference before the cue start
_C.DATA.MEAN = 0
_C.DATA.STD = 25
_C.DATA.TIME_WIN = [0.5, 3.5]
_C.DATA.DUR = 3
_C.DATA.WIN_STEP = 4

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'IFNet'
_C.MODEL.NUM_CLASSES = 4
_C.MODEL.TIME_POINTS = int(_C.DATA.DUR * int(_C.DATA.FS / _C.DATA.RESAMPLE))
_C.MODEL.IN_CHANS = 22
_C.MODEL.PATCH_SIZE = 125   #temporal pooling size
_C.MODEL.EMBED_DIMS = 64
_C.MODEL.KERNEL_SIZE = 63
_C.MODEL.RADIX = 2

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.DEVICE = 0
_C.SEED = 618
_C.LOG = False
_C.SAVE = False
_C.EVAL = False
_C.EVAL_TAG = ''
_C.OUTPUT = 'output'
_C.TAG = 'BCIC_IFNet'    #file name to log the experiment process and results

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 1000
_C.TRAIN.BASE_LR = 2 ** -12
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.LR_SCHEDULER = None

_C.TRAIN.RETRAIN = True
_C.TRAIN.RETRAIN_EPOCHS = 500

_C.TRAIN.OPTIMIZER = 'AdamW'
_C.TRAIN.CRITERION = 'CE'
_C.TRAIN.REPEAT = 2  #repeat training with differnt network initilizations


def get_config():
    return _C.clone()