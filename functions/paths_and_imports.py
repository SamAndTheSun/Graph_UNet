# Standard library imports
import os
import warnings
import subprocess

# Third-party imports
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import torch
from scipy.io import loadmat

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Statistical analysis
import statsmodels as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

# Suppress warnings
warnings.filterwarnings('ignore')  # Explanation: pytorch warning when converting numpy array slice to tensor

# Local function imports
from nn_optim_unet import *
from postprocessing import *

# Local variables
from datasets_config import datasets

# ico lvl
ico_levels=[6, 5, 4]
starting_ico = ico_levels[0] # representing starting ico, i.e. highest ico

# Which hemi is first index wise
first = 'rh'

# dir with data
data_dir = '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/processed/'

# Where to put outputs
output_dir = '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/last_model_outputs/'

# Where the pooling data is
pooling_path='/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/pooling/'

# Paths for training data

X_pretrain = f'{data_dir}X_pretrain.npy'
y_pretrain = f'{data_dir}y_pretrain.npy'

X_train = f'{data_dir}X_train.npy'
y_train = f'{data_dir}y_train.npy'

X_test_CN = f'{data_dir}X_ADNI_CN.npy'
y_test_CN = f'{data_dir}y_ADNI_CN.npy'

X_test_AD = f'{data_dir}X_ADNI_AD.npy'
y_test_AD = f'{data_dir}y_ADNI_AD.npy'

X_test_CN_female = f'{data_dir}X_ADNI_CN_F.npy'
y_test_CN_female = f'{data_dir}y_ADNI_CN_F.npy'

X_test_CN_male = f'{data_dir}X_ADNI_CN_M.npy'
y_test_CN_male = f'{data_dir}y_ADNI_CN_M.npy'

# Training parameters

batch_size = 128
batch_load = 128
n_pretrain_epochs = 25
n_train_epochs = 50
lr = 0.01 
weight_decay = 0

intra_w = 0.5
global_w = 1
feature_scale = 1
dropout_levels = [0, 0, 0.5, 0.5, 0]

print_every = 25