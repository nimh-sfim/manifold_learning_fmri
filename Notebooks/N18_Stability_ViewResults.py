# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Embeddings2 + Sdim
#     language: python
#     name: embeddings3
# ---

# # Description
#
# This notebook generates the figures summarizing the results of the stability analysis for T-SNE and UMAP embeddings.

import pandas as pd
import os.path as osp
import numpy as np
import hvplot.pandas
from utils.basics import PRJ_DIR, PNAS2015_subject_list
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration variables common to both techniques

norm_method = 'asis'
wls = 45
wss = 1.5
N_iters = 1000

# Configuration variable for T-SNE (Best set of hyper-parameters)

tsne_init_method = 'pca'
tsne_dist = 'correlation'
tsne_pp = 65
tsne_m = 2
tsne_alpha = 10

# Configuration variables for UMAP (Best set of hyper-parameters)

umap_m = 3
umap_knn = 70
umap_dist = 'euclidean'
umap_alpha = 0.01
umap_min_dist = 0.8
umap_init_method = 'spectral'

# # 1. Load SI task values for T-SNE

# %%time
df_tsne = pd.DataFrame(columns=['Subject','N_iter','SI'])
for subject in PNAS2015_subject_list:
    for n_iter in range(N_iters):
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE','Stability','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.SI.I{n_iter}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=tsne_init_method,
                                                                                                                                                   dist=tsne_dist,
                                                                                                                                                   pp=str(tsne_pp).zfill(4),
                                                                                                                                                   m=str(tsne_m).zfill(4),
                                                                                                                                                   lr=str(tsne_alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
        aux_si = pd.read_pickle(path)
        df_tsne = df_tsne.append({'Subject':subject,'N_iter':n_iter,'SI':aux_si.values[0][0]}, ignore_index=True)

# # 2. Load SI task values for UMAP

# %%time
df_umap = pd.DataFrame(columns=['Subject','N_iter','SI'])
for subject in PNAS2015_subject_list:
    for n_iter in range(N_iters):
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'UMAP','Stability','{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.SI.I{n_iter}.pkl'.format(norm_method=norm_method, init_method=umap_init_method,
                                                                                                                                                   sbj=subject,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=umap_dist,
                                                                                                                                                   knn=str(umap_knn).zfill(4),
                                                                                                                                                   m=str(umap_m).zfill(4),
                                                                                                                                                   min_dist=str(umap_min_dist),
                                                                                                                                                   alpha=str(umap_alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
        aux_si = pd.read_pickle(path)
        df_umap = df_umap.append({'Subject':subject,'N_iter':n_iter,'SI':aux_si[umap_m].values[0]}, ignore_index=True)

# # 3. Plot results

fig,axs = plt.subplots(1,2,figsize=(20,5))
# T-SNE Results
sns.set(font_scale=1.5)
plot = sns.boxplot(data=df_tsne,y='SI', x='Subject', ax=axs[0])
plot.set_ylabel('$SI_{task}$');
plot.set_title('T-SNE')
plot.set_xticklabels(PNAS2015_subject_list,rotation=90);
#plt.xticks(rotation=90);
# UMAP Results
plot = sns.boxplot(data=df_umap,y='SI', x='Subject')
plot.set_ylabel('$SI_{task}$')
plot.set_title('UMAP')
plot.set_xticklabels(PNAS2015_subject_list,rotation=90);
plt.tight_layout()

fig.savefig('../Outputs/Sup_Figures/Supp_Figure05.png')
