# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: opentsne
#     language: python
#     name: opentsne
# ---

# # DESCRIPTION: Supplementary Figure 6
#
# This notebook generates Supplementary Figure 6

import pandas as pd
import hvplot.pandas
import os.path as osp
import numpy as np
from utils.basics import PRJ_DIR, PNAS2015_subject_list, sbj_cmap_dict, sbj_cmap_list
label_dict = {'SBJ06':'Sbj 1','SBJ07':'Sbj 2','SBJ08':'Sbj 3','SBJ09':'Sbj 4','SBJ10':'Sbj 5','SBJ11':'Sbj 6','SBJ12':'Sbj 7','SBJ13':'Sbj 8','SBJ16':'Sbj 9','SBJ17':'Sbj 10',
              'SBJ18':'Sbj 11','SBJ19':'Sbj 12','SBJ20':'Sbj 13','SBJ21':'Sbj 14','SBJ22':'Sbj 15','SBJ23':'Sbj 16','SBJ24':'Sbj 17','SBJ25':'Sbj 18','SBJ26':'Sbj 19','SBJ27':'Sbj 20'}

# ***
#
# # Non-normalized Data

# For each scan, we do the following:
#     
# 1) Load its tvFC matrix
# 2) Compute the mean of tvFC timeseries per connection
# 3) Compute the stdv of tvFC timeseries per connection
# 4) Store these values in a dataframe
#
# > **NOTE:** We relabel the columns to have the final scan/subject ids for plotting purposes.

# +
# %%time
df_asis_mean = None
df_asis_stdv = None
for sbj in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Original','{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.asis.pkl'.format(sbj=sbj))
    df   = pd.read_pickle(path)
    if df_asis_mean is None:
        df_asis_mean = df.mean(axis=1).reset_index(drop=True)
        df_asis_mean = pd.DataFrame(df_asis_mean, columns=[sbj])
        df_asis_stdv = df.std(axis=1).reset_index(drop=True)
        df_asis_stdv = pd.DataFrame(df_asis_stdv, columns=[sbj])
    else:
        aux_mean = df.mean(axis=1).reset_index(drop=True)
        aux_mean = pd.DataFrame(aux_mean, columns=[sbj])
        df_asis_mean = pd.concat([df_asis_mean,aux_mean],axis=1)
        aux_stdv = df.std(axis=1).reset_index(drop=True)
        aux_stdv = pd.DataFrame(aux_stdv, columns=[sbj])
        df_asis_stdv = pd.concat([df_asis_stdv,aux_stdv],axis=1)

df_asis_mean.columns = [label_dict[s] for s in df_asis_mean.columns]
df_asis_stdv.columns = [label_dict[s] for s in df_asis_stdv.columns]
# -

# We plot the distributions of mean and standard deviations across all connections

df_asis_stdv.hvplot.violin(cmap=sbj_cmap_list, xlabel='Subject', ylabel='Stdv of tvFC').opts(xrotation=45) + df_asis_mean.hvplot.violin(cmap=sbj_cmap_list, xlabel='Subject', ylabel='Mean of tvFC', shared_axes=False).opts(xrotation=45)

# ***
#
# # Normalized Data
#
# We do the same operation as above, but this time we load the zscored version of the matrices

# %%time
df_zscored_mean = None
df_zscored_stdv = None
for sbj in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Original','{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.zscored.pkl'.format(sbj=sbj))
    df   = pd.read_pickle(path)
    if df_zscored_mean is None:
        df_zscored_mean = df.mean(axis=1).reset_index(drop=True)
        df_zscored_mean = pd.DataFrame(df_zscored_mean, columns=[sbj])
        df_zscored_stdv = df.std(axis=1).reset_index(drop=True)
        df_zscored_stdv = pd.DataFrame(df_zscored_stdv, columns=[sbj])
    else:
        aux_mean = df.mean(axis=1).reset_index(drop=True)
        aux_mean = pd.DataFrame(aux_mean, columns=[sbj])
        df_zscored_mean = pd.concat([df_zscored_mean,aux_mean],axis=1)
        aux_stdv = df.std(axis=1).reset_index(drop=True)
        aux_stdv = pd.DataFrame(aux_stdv, columns=[sbj])
        df_zscored_stdv = pd.concat([df_zscored_stdv,aux_stdv],axis=1)
df_zscored_mean.columns = [label_dict[s] for s in df_zscored_mean.columns]
df_zscored_stdv.columns = [label_dict[s] for s in df_zscored_stdv.columns]

# We plot the distributions

df_zscored_stdv.hvplot.violin(cmap=sbj_cmap_list, xlabel='Subject', ylabel='Stdv of tvFC').opts(xrotation=45) + df_zscored_mean.hvplot.violin(cmap=sbj_cmap_list, xlabel='Subject', ylabel='Mean of tvFC', shared_axes=False).opts(xrotation=45)
