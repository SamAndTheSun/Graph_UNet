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
from matplotlib.patches import Rectangle
import seaborn as sns

# Statistical analysis
import statsmodels as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, spearmanr, rankdata, norm
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression

# Suppress warnings
warnings.filterwarnings('ignore')  # Explanation: pytorch warning when converting numpy array slice to tensor

# Local function imports
from nn_optim_unet import *
from postprocessing import *

# Local variables
from dataset_config import datasets

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

# Where cognitive data is
cog_path = '/mnt/md0/tempFolder/samAnderson/datasets/ADNI_cognitive_scores.csv'

# Paths for training data

X_pretrain = f'{data_dir}X_pretrain.npy'
y_pretrain = f'{data_dir}y_pretrain.npy'

X_train = f'{data_dir}X_train.npy'
y_train = f'{data_dir}y_train.npy'
sex_train = f'{data_dir}sex_train.npy'

X_test_CN = f'{data_dir}X_ADNI_CN.npy'
y_test_CN = f'{data_dir}y_ADNI_CN.npy'

X_test_AD = f'{data_dir}X_ADNI_AD.npy'
y_test_AD = f'{data_dir}y_ADNI_AD.npy'

# Training parameters

batch_size = 128
batch_load = 128
n_train_epochs = 50
lr = 0.01 
weight_decay = 0

intra_w = 0.5
global_w = 1
feature_scale = 1
dropout_levels = [0, 0, 0.5, 0.5, 0]

print_every = 25

# Create a dictionary indicating whether higher scores are associated with better performance

test_relations = {
    'ADAS11' : False,            # Alzheimer's Disease Assessment Scale - Cognitive Subscale (11 items)
    'ADAS13' : False,            # Alzheimer's Disease Assessment Scale - 13-item version
    'ADASQ4' : False,            # Subcomponent of ADAS
    'CDRSB' : False,             # Clinical Dementia Rating - Sum of Boxes
    'DIGITSCOR' : True,          # Digit Span (forward/backward) - higher = better
    'EcogPtDivatt' : False,      # ECog Patient: Divided Attention - higher = more impairment
    'EcogPtLang' : False,        # ECog Patient: Language
    'EcogPtMem' : False,         # ECog Patient: Memory
    'EcogPtOrgan' : False,       # ECog Patient: Organization
    'EcogPtPlan' : False,        # ECog Patient: Planning
    'EcogPtVisspat' : False,     # ECog Patient: Visuospatial
    'EcogPtTotal': False,  # ECog Patient Total Score – higher values indicate greater self-reported cognitive impairment
    'EcogSPDivatt' : False,      # ECog Study Partner: Divided Attention
    'EcogSPLang' : False,        # ECog Study Partner: Language
    'EcogSPMem' : False,         # ECog Study Partner: Memory
    'EcogSPOrgan' : False,       # ECog Study Partner: Organization
    'EcogSPPlan' : False,        # ECog Study Partner: Planning
    'EcogSPTotal' : False,       # ECog Study Partner: Total Score
    'EcogSPVisspat' : False,     # ECog Study Partner: Visuospatial
    'FAQ' : False,               # Functional Activities Questionnaire - higher = more impairment
    'LDELTOTAL' : True,          # Logical Memory Delayed Recall - higher = better
    'MMSE' : True,               # Mini-Mental State Examination
    'MOCA' : True,               # Montreal Cognitive Assessment
    'RAVLT_forgetting' : False,  # Rey Auditory Verbal Learning Test - higher = worse retention
    'RAVLT_immediate' : True,    # RAVLT immediate recall
    'RAVLT_learning' : True,     # RAVLT learning score
    'RAVLT_perc_forgetting' : False,  # Percent forgetting - higher = worse
    'TRABSCOR' : False           # Trail Making Test Part B - Score = time, higher = worse
}