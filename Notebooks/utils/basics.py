import pandas as pd
import numpy as np
import os.path as osp

from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex, rgb_to_hsv

PRJ_DIR = '/data/SFIMJGC_HCP7T/manifold_learning_fmri'

task_cmap      = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'pink'}
task_cmap_caps = {'REST': 'gray', 'BACK': 'blue',   'VIDE':  '#F4D03F',  'MATH': 'green', 'XXXX': 'pink'}

PNAS2015_folder         = '/data/SFIMJGC/PRJ_CognitiveStateDetection01/'
PNAS2015_subject_list   = ['SBJ06', 'SBJ07', 'SBJ08', 'SBJ09', 'SBJ10', 'SBJ11', 'SBJ12', 'SBJ13', 'SBJ16', 'SBJ17', 'SBJ18', 'SBJ19', 'SBJ20', 'SBJ21', 'SBJ22', 'SBJ23', 'SBJ24', 'SBJ25', 'SBJ26', 'SBJ27']
PNAS2015_roi_names_path = osp.join(PRJ_DIR,'Resources/PNAS2015_ROI_Names.txt')
PNAS2015_win_names_paths = {(45,1.5): '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Resources/PNAS2015_WinNames_wl45s_ws1p5s.txt'}

sbj_cmap_list = [rgb2hex(c) for c in get_cmap('tab20',20).colors]
sbj_cmap_dict = {PNAS2015_subject_list[i]:sbj_cmap_list[i] for i in range(len(PNAS2015_subject_list))}

input_datas  = ['Original','Null_ConnRand','Null_PhaseRand']
norm_methods = ['asis','zscored']

group_method_2_label = {'ALL':'Concatenation','Procrustes':'Procrustes'}

# Laplacian Eigenmap Variables
le_dist_metrics = ['euclidean','correlation','cosine']
le_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)]
le_ms           = [2,3,5,10,15,20,25,30]

# TSNE Variables
tsne_dist_metrics = ['euclidean','correlation','cosine']
#tsne_pps          = [int(i) for i in np.linspace(start=5, stop=100, num=20)] + [125, 150, 175, 200]
tsne_pps          = [int(i) for i in np.linspace(start=20, stop=100, num=17)] + [125, 150, 175, 200]
tsne_ms           = [2,3,5,10,15,20,25,30]
tsne_alphas       = [10, 50, 75, 100, 200, 500, 1000]
tsne_inits        = ['pca']

# UMAP Variables
umap_dist_metrics = ['euclidean','correlation','cosine']
umap_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)]
umap_ms           = [2,3,5,10,15,20,25,30]
umap_alphas       = [0.01, 0.1, 1.0]
umap_inits        = ['spectral']

def load_representative_tvFC_data():
    print('++ INFO: Loading the tvFC dataset.....')
    X_df = pd.read_csv('../Resources/Figure03/swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv.gz', index_col=[0,1])
    # Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
    X_df.columns = X_df.columns.str.split('.').str[0]
    return X_df