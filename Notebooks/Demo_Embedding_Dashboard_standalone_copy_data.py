# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: opentsne_panel14
#     language: python
#     name: opentsne_panel14
# ---

# # Description
#
# The stand-alone dashboards generated for this project (which are intended only for demo purposes) need data to be accessible via a URL. To accompish this we created a separate github repo (manifold_learning_fmri_demo_data) to hold a subset of the data.
#
# This notebook can be used to copy / add more demo data to this secondary repo. Once data is copied you will need to add files via ```git add```, commit ```git commit``` and finally push to github ```git push```.

# +
import panel as pn
import numpy as np
import os.path as osp
import pandas as pd
from scipy.stats import zscore
from matplotlib.colors import rgb2hex
import matplotlib

import hvplot.pandas
import plotly.express as px
# -

from tqdm import tqdm
import os

# So far we are working with these values of wls and wss across the whole manuscript
wls = 45
wss = 1.5
min_dist = 0.8

# +
# Available scans
avail_scans_dict = {'Scan 1':'SBJ06', 'Scan 2':'SBJ07'}

# Available Data Scenarios
input_data_dict = {'Real Data':'Original','Connectivity Randomization':'Null_ConnRand','Phase Randomization':'Null_PhaseRand'}

# Normalization Options
normalization_dict = {'Do not normalize':'asis','Z-score':'zscored'}

# Colormaps
#sbj_cmap_list = [rgb2hex(c) for c in matplotlib.colormaps['tab20'].colors]
# Hard coded below to avoid importing matplotlib
sbj_cmap_list = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5']
sbj_cmap = {v:sbj_cmap_list[i] for i,v in enumerate(avail_scans_dict.values())}
task_cmap = {'REST': 'gray', 'BACK': 'blue',   'VIDE':  '#F4D03F',  'MATH': 'green', 'XXXX': 'pink'}

# Laplacian Eigenmap related options
le_dist_metrics = {'Euclidena Distance':'euclidean','Correlation Distance':'correlation','Cosine Distance':'cosine'}
le_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)][::5]
le_ms           = [2,3,5,10,15]

# UMAP related options
umap_dist_metrics = le_dist_metrics
umap_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)][::5]
umap_ms           = [2,3,5,10]
umap_alphas       = [0.01, 1.0]
umap_inits        = ['spectral']

# T-SNE related options
tsne_dist_metrics = le_dist_metrics
tsne_pps          = [5,50,100,150,200]
tsne_ms           = [2,3,5,10]
tsne_alphas       = [10, 50, 100, 1000]
tsne_inits        = ['pca']
# -

IN_DIR  = osp.join('/data','SFIMJGC_HCP7T','manifold_learning_fmri','Data_Interim','PNAS2015')
OUT_DIR =  osp.join('/data','SFIMJGC_HCP7T','manifold_learning_fmri_demo_data','data','embeddings')

# ### Copy LE demo files to demo repo

for sbj in list(avail_scans_dict.values()):
    for input_data in tqdm(list(input_data_dict.values())):
        for scenario in list(normalization_dict.values()):
            for dist in list(le_dist_metrics.values()):
                for knn in le_knns:
                    for m in le_ms:
                        input_path = osp.join(IN_DIR,sbj,'LE',input_data,
                            '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(sbj=sbj,scenario=scenario,wls=str(int(wls)).zfill(3),wss=str(wss),
                                                                                                        dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4)))
                        output_folder = osp.join(OUT_DIR,sbj,'LE',input_data)
                        output_file   = osp.join(output_folder,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(sbj=sbj,scenario=scenario,wls=str(int(wls)).zfill(3),wss=str(wss),
                                                                                                        dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4)))
                        df = pd.read_pickle(input_path)
                        if not osp.exists(output_folder):
                            os.makedirs(output_folder)
                        df.to_pickle(output_file)

# ### Copy TSNE demo files to demo repo

init_method = 'pca'
for sbj in list(avail_scans_dict.values()):
    for input_data in tqdm(list(input_data_dict.values())):
        for scenario in list(normalization_dict.values()):
            for dist in list(tsne_dist_metrics.values()):
                for pp in tsne_pps:
                    for m in tsne_ms:
                        for alpha in tsne_alphas:                    
                            input_path = osp.join(IN_DIR,sbj,'TSNE',input_data,
                                '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,init_method=init_method,sbj=sbj,
                                                                                                                                                       wls=str(int(wls)).zfill(3), 
                                                                                                                                                       wss=str(wss),
                                                                                                                                                       dist=dist,
                                                                                                                                                       pp=str(pp).zfill(4),
                                                                                                                                                       m=str(m).zfill(4),
                                                                                                                                                       alpha=str(alpha)))
                            output_folder = osp.join(OUT_DIR,sbj,'TSNE',input_data)
                            output_file   = osp.join(output_folder,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,init_method=init_method,sbj=sbj,
                                                                                                                                                       wls=str(int(wls)).zfill(3), 
                                                                                                                                                       wss=str(wss),
                                                                                                                                                       dist=dist,
                                                                                                                                                       pp=str(pp).zfill(4),
                                                                                                                                                       m=str(m).zfill(4),
                                                                                                                                                       alpha=str(alpha)))
                            df = pd.read_pickle(input_path)
                            if not osp.exists(output_folder):
                                os.makedirs(output_folder)
                            df.to_pickle(output_file)

# ### Copy UMAP demo files to demo repo

for sbj in list(avail_scans_dict.values()):
    for input_data in tqdm(list(input_data_dict.values())):
        for scenario in list(normalization_dict.values()):
            for dist in list(tsne_dist_metrics.values()):
                for knn in umap_knns:
                    for m in umap_ms:
                        for alpha in umap_alphas:
                            for init_method in umap_inits:
                                input_path = osp.join(IN_DIR,sbj,'UMAP',input_data,
                                '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,        
                                                                                                init_method=init_method,sbj=sbj,wls=str(int(wls)).zfill(3),wss=str(wss),
                                                                                                dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4),min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                                output_folder = osp.join(OUT_DIR,sbj,'UMAP',input_data)
                                output_file   = osp.join(output_folder,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,        
                                                                                                init_method=init_method,sbj=sbj,wls=str(int(wls)).zfill(3),wss=str(wss),
                                                                                                dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4),min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                                df = pd.read_pickle(input_path)
                                if not osp.exists(output_folder):
                                    os.makedirs(output_folder)
                                df.to_pickle(output_file)
