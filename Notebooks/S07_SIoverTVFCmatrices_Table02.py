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

# # Description - Table 02 SI using tvFC in original space
#
# This notebook will compute the Silhouette Index when using as input tvFC matrices without any dimensionality reduction.
#
# * SItask at the scan level --> one value per subject --> we report mean and stdev
#
# * SIsbj at the group level --> only one value available
#
# We do this using as input both the normalized and non-normalized matrices

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from utils.basics import PRJ_DIR, PNAS2015_subject_list
import os.path as osp
from tqdm import tqdm

norm_methods = ['asis','zscored']

# # 1. Scan level embeddings (SI task)

# %%time
df = pd.DataFrame(columns=['Subject','Norm Method','SI'])
for sbj in tqdm(PNAS2015_subject_list, desc='Subject'):
    for norm_method in norm_methods:
        input_path      = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Original',f'{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.{norm_method}.pkl')
        data            = pd.read_pickle(input_path).T
        data.index.name = 'Task'
        data            = data.drop('XXXX') 
        input_data      = data.values
        input_labels    = list(data.index)
        si_value        = silhouette_score(input_data, input_labels, n_jobs=-1)
        df              = df.append({'Subject':sbj,'Norm Method':norm_method,'SI':si_value}, ignore_index=True)

# Here, we print the mean SI across all subjects.

df.groupby(by='Norm Method').mean().round(2)

# Next, we print the standard deviation across all subjects

df.groupby(by='Norm Method').std().round(2)

# # 2. Group Level SI
#
# ## 2.1. Load tvFC matrices for all subjects and concatenate them on a single large tvFC group matrix

# +
# %%time
# Create empty dictionary that will hold the group level tvFC matrices
group_data = {'asis':None,
              'zscored':None}

for norm_method in norm_methods:
    for sbj in tqdm(PNAS2015_subject_list, desc='Subject'):
        input_path      = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Original',f'{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.{norm_method}.pkl')
        data            = pd.read_pickle(input_path).T
        data.index.name = 'Task'
        data            = data.drop('XXXX') 
        data            = data.reset_index()
        data['Subject'] = sbj
        data            = data.set_index(['Subject','Task'])
        if group_data[norm_method] is None:
            group_data[norm_method] = data
        else:
            group_data[norm_method] = pd.concat([group_data[norm_method], data],axis=0)
# -

# ## 2.2. Compute SI using the group level tvFC matrices

# %%time
for norm_method in norm_methods:
    print(f'++ INFO: Results for {norm_method} matrices')
    input_labels_sbj = list(group_data[norm_method].index.get_level_values('Subject'))
    input_labels_tsk = list(group_data[norm_method].index.get_level_values('Task'))
    input_data       = group_data[norm_method].values
    si_value_sbj     = silhouette_score(input_data, input_labels_sbj,n_jobs=-1)
    si_value_tsk     = silhouette_score(input_data, input_labels_tsk,n_jobs=-1)
    print('++ INFO: Group Level SItask    [%s] = %.02f' % (norm_method,si_value_tsk))
    print('++ INFO: Group Level SIsubject [%s] = %.02f' % (norm_method,si_value_sbj))
