# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Embeddings2 + Sdim
#     language: python
#     name: embeddings3
# ---

import pandas as pd
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from scipy.spatial.distance import euclidean   as euclidean_distance
from scipy.spatial.distance import cosine      as cosine_distance
from scipy.spatial.distance import correlation as correlation_distance
from scipy.stats import pearsonr as correlation

from utils.basics import PRJ_DIR

df   = pd.DataFrame([[.5,1],[7.15,1],[2,4],[.5,4]],columns=['x','y'],index=['a','b','c','d'])
df.index.name = 'Sample'
df.reset_index(inplace=True)
df.index = df['Sample']

points = hv.Points(df,kdims=['x','y']).opts(size=10,color='k',fontsize={'xlabel':12,'ylabel':12,'ticks':12}, xlim=(0,8), ylim=(0,4.5), width=800, height=400)
labels = hv.Labels(df,kdims=['x','y'],vdims=['Sample'])

(points * labels).opts(opts.Labels(xoffset=0.2, yoffset=0.2))

df_dist=pd.DataFrame(columns=dmetric_list, index=pd.MultiIndex.from_tuples([('a','b'),('a','c'),('a','d'),('b','c'),('b','d'),('c','d')], names=['i','j']))
for i,j in df_dist.index:
    df_dist.loc[(i,j),'euclidean']   = euclidean_distance(df.loc[i,['x','y']],df.loc[j,['x','y']])
    df_dist.loc[(i,j),'cosine']      = cosine_distance(df.loc[i,['x','y']],df.loc[j,['x','y']])
    df_dist.loc[(i,j),'correlation'] = correlation_distance(df.loc[i,['x','y']].T,df.loc[j,['x','y']].T)
df_dist = df_dist.infer_objects()

df_dist.round(2)

dict_dist = {}
for dist in dmetric_list:
    dict_dist[dist] = pd.DataFrame(index=['a','b','c','d'], columns=['a','b','c','d'])
    for i in ['a','b','c','d']:
        for j in ['a','b','c','d']:
            if i == j:
                dict_dist[dist].loc[i,j] = 0
            else:
                try:
                    dict_dist[dist].loc[i,j] = df_dist.loc[(i,j),dist]
                except:
                    dict_dist[dist].loc[i,j] = df_dist.loc[(j,i),dist]
    dict_dist[dist] = dict_dist[dist].infer_objects()
    dict_dist[dist].name = dist
    dict_dist[dist] = dict_dist[dist].round(2)

# +
deuc_heatmap = dict_dist['euclidean'].T.hvplot.heatmap(aspect='square', cmap='viridis', fontsize={'ticks':13, 'clabel':13}, clabel='Euclidean Distance:', clim=(0,7.5))
dcor_heatmap = dict_dist['correlation'].T.hvplot.heatmap(aspect='square', cmap='viridis', fontsize={'ticks':13, 'clabel':13}, clabel='Correlation Distance:', clim=(0,2))
dcos_heatmap = dict_dist['cosine'].T.hvplot.heatmap(aspect='square', cmap='viridis', fontsize={'ticks':13, 'clabel':13}, clabel='Cosine Distance:', clim=(0,2))

(deuc_heatmap*hv.Labels(deuc_heatmap)).opts(opts.Labels(text_color='w'))
# -

(dcor_heatmap*hv.Labels(dcor_heatmap)).opts(opts.Labels(text_color='w'))

(dcos_heatmap*hv.Labels(dcos_heatmap)).opts(opts.Labels(text_color='w'))

# +
dmetric_list = ['euclidean','correlation','cosine']
dmetric_mins = [0,0,0]
dmetric_maxs = [65,1.5,1.5]
# Cosine Similarity (CS) and Cosine Distance (CD)
#  - Angle is zero        --> CS = 1  --> CD = 1 - 1  = 0
#  - Angle is 90 or 270   --> CS = 0  --> CD = 1 - 0  = 1
#  - Angle is 180         --> CS = -1 --> CD = 1 - -1 = 2

# Correlation Distance = 1 - Pearson Correlation | r [1, 0, -1] --> cor_dis [0, 1, 2]
# -

np.cos(np.radians(270))

# ### Load SWC Matrix

swc_path    = osp.join(PRJ_DIR,'Resources','Figure03','swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv')
swc         = pd.read_csv(swc_path, index_col=[0,1], header=0)
# Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
swc.columns = swc.columns.str.split('.').str[0]

print('++ INFO: Size of SWC dataframe is %d connections X %d windows.' % swc.shape)
swc.head(5)

fig = plt.figure(figsize=(40,5))
sns.heatmap(swc, cmap='gray')

# ### Drop In-between windows

swc        = swc.drop('XXXX',axis=1)
win_labels = swc.columns

print('++ INFO: Size of SWC dataframe is %d connections X %d windows.' % swc.shape)
fig = plt.figure(figsize=(40,5))
sns.heatmap(swc, cmap='gray')

# ### Create Dissimilarity Matrix

from scipy.spatial.distance import pdist, squareform

# %%time
dist_matrix = {}
for dmetric in dmetric_list:
    dist_matrix[dmetric] = squareform(pdist(swc.T, dmetric))

fig,axs = plt.subplots(1,3,figsize=(24,5))
for idx,(dm, vmin, vmax) in enumerate(zip(dmetric_list,dmetric_mins,dmetric_maxs)):
    sns.heatmap(dist_matrix[dm], cmap='viridis', vmin=vmin, vmax=vmax, ax=axs[idx], square=True)
    axs[idx].yaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
    axs[idx].xaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
    axs[idx].yaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO']);
    axs[idx].xaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO']);


