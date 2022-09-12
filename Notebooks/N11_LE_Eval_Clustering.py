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

# This notebook creates a summary view of the clustering-based quality of LE embeddings for all data inputs (e.g., real and nullified) across all considered scenarios (e.g., m, knn, dist).

import pandas as pd
import xarray as xr
import numpy as np
import os.path as osp
import hvplot.pandas
import hvplot.xarray
import seaborn as sns
import matplotlib.pyplot as plt
from utils.basics import PNAS2015_subject_list, PRJ_DIR
from utils.basics import le_dist_metrics, le_knns, le_ms
from utils.basics import task_cmap_caps as task_cmap
from scipy.stats import zscore
import panel as pn
import plotly.express as px
from IPython.display import Image
from tqdm.notebook import tqdm

from sklearn.metrics import silhouette_score
from statannotations.Annotator import Annotator
from scipy.spatial import procrustes

# So far we are working with these values of wls and wss across the whole manuscript
wls = 45
wss = 1.5

# ### Load Real Data
#
# The following cells create two data structures that we will use throughout the rest of the notebook
#
# * ```si_xrs```: dictionary of xr.DataArray objects. There will be one entry per data type. Each xr.DataArray object contains the SI for all combinations of subjects, distances, knn and m values. 
# * ```embs_xrs```:  dictionary of dictionaries. In each entry there is a dictionary containing one dataframe per combination of subjects, distances, knn and m values. The dataframe contains the particular embedding associated with that combination of hyper-parameters
#
# First, we create empty versions of the dictionary structures

si_xrs = {'Original':                 xr.DataArray(dims=['subject','dist','knn','m'],coords={'subject':PNAS2015_subject_list, 'dist':le_dist_metrics,'knn':le_knns,'m':[2,3]}),
          'Connection Randomization': xr.DataArray(dims=['subject','dist','knn','m'],coords={'subject':PNAS2015_subject_list, 'dist':le_dist_metrics,'knn':le_knns,'m':[2,3]}),
          'Phase Randomization':      xr.DataArray(dims=['subject','dist','knn','m'],coords={'subject':PNAS2015_subject_list, 'dist':le_dist_metrics,'knn':le_knns,'m':[2,3]})}

embs_xrs = {'Original':                 {},
          'Connection Randomization': {},
          'Phase Randomization':      {}}

# Second, we populate the part corresponding to the original data

# + tags=[]
# %%time
for m in tqdm([2,3], desc='Final Dimensions', position=0, leave=True):
    for dist in tqdm(le_dist_metrics, desc='Distance Metric', leave=False):
        for knn in le_knns:
            for sbj in PNAS2015_subject_list:
                path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE','{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.pkl'.format(sbj=sbj,
                                                                                                                                         wls=str(int(wls)).zfill(3), 
                                                                                                                                         wss=str(wss),
                                                                                                                                         dist=dist,
                                                                                                                                         knn=str(knn).zfill(4),
                                                                                                                                         m=str(m).zfill(4)))
                if not osp.exists(path):
                    print('++ WARNING: Missing file: %s' % path)
                    continue
                df = pd.read_pickle(path)
                df_pure = df.drop('XXXX').copy()
                si_xrs['Original'].loc[sbj,dist,knn,m]=silhouette_score(df_pure, df_pure.index)
                embs_xrs['Original'][(sbj,dist,knn,m)]=df
# -

# Third, we populate the part corresponding to the connection randomized data

# %%time
for m in tqdm([2,3], desc='Final Dimensions', position=0, leave=True):
    for dist in tqdm(le_dist_metrics, desc='Distance Metric', leave=False):
        for knn in le_knns:
            for sbj in PNAS2015_subject_list:
                path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE','Null_ConnRand','{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.pkl'.format(sbj=sbj,
                                                                                                                                         wls=str(int(wls)).zfill(3), 
                                                                                                                                         wss=str(wss),
                                                                                                                                         dist=dist,
                                                                                                                                         knn=str(knn).zfill(4),
                                                                                                                                         m=str(m).zfill(4)))
                if not osp.exists(path):
                    print('++ WARNING: Missing file: %s' % path)
                    continue
                df = pd.read_pickle(path)
                df_pure = df.drop('XXXX').copy()
                si_xrs['Connection Randomization'].loc[sbj,dist,knn,m]=silhouette_score(df_pure, df_pure.index)
                embs_xrs['Connection Randomization'][(sbj,dist,knn,m)]=df

# Forth, we populte the part corresponding to the phase randmized data

# %%time
for m in tqdm([2,3], desc='Final Dimensions', position=0, leave=True):
    for dist in tqdm(le_dist_metrics, desc='Distance Metric', leave=False):
        for knn in le_knns:
            for sbj in PNAS2015_subject_list:
                path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE','Null_PhaseRand','{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.pkl'.format(sbj=sbj,
                                                                                                                                         wls=str(int(wls)).zfill(3), 
                                                                                                                                         wss=str(wss),
                                                                                                                                         dist=dist,
                                                                                                                                         knn=str(knn).zfill(4),
                                                                                                                                         m=str(m).zfill(4)))
                if not osp.exists(path):
                    print('++ WARNING: Missing file: %s' % path)
                    continue
                df = pd.read_pickle(path)
                df_pure = df.drop('XXXX').copy()
                si_xrs['Phase Randomization'].loc[sbj,dist,knn,m]=silhouette_score(df_pure, df_pure.index)
                embs_xrs['Phase Randomization'][(sbj,dist,knn,m)]=df

# ***
# # Panel A. SI for all data types, knn, metrics and m=2,3
#
# The next cell takes ```si_xrs``` and seaborn to plot the evolution of SI across knn for the different metric types and also as a function of whether the input is the original data or data that has been nullified.

aux_df = si_xrs['Original'].to_dataframe(name='SI').reset_index()
sns.set(font_scale=2)
fig,ax = plt.subplots(1,1,figsize=(10,7))
g = sns.lineplot(data=aux_df,x='knn',y='SI', hue='dist', palette=['red','green','blue'], hue_order=['euclidean','cosine','correlation'], style='m', style_order=[3,2], ax=ax)
g.set_xlim(30,200)
g.set_ylim(-.1,.6)
g.set_xlabel('Neighborhood Size [knn]')
g.set_ylabel('Silhouette Index')
g.legend(loc='lower left', ncol=2)
g.set_xticks([30,65,100,135,170])
g.yaxis.tick_right()
g.grid(True)
aux_df = si_xrs['Connection Randomization'].to_dataframe(name='SI').reset_index()
sns.lineplot(data=aux_df,x='knn',y='SI', hue='dist', palette=['black','black','black'], hue_order=['euclidean','cosine','correlation'], style='m', style_order=[3,2], ax=ax, legend=None)
aux_df = si_xrs['Phase Randomization'].to_dataframe(name='SI').reset_index()
sns.lineplot(data=aux_df,x='knn',y='SI', hue='dist', palette=['grey','grey','grey'], hue_order=['euclidean','cosine','correlation'], style='m', style_order=[3,2], ax=ax, legend=None)

# The next two cells show the same information as the one above, but in three separate graphs, one per input data type (original, conn rand, phase rand)

# %%time
si_vs_knn_plots = {}
for data_input in ['Original','Connection Randomization','Phase Randomization']:
    aux_df = si_xrs[data_input].to_dataframe(name='SI').reset_index()
    sns.set(font_scale=2)
    fig,ax = plt.subplots(1,1,figsize=(10,7))
    g = sns.lineplot(data=aux_df,x='knn',y='SI', hue='dist', palette=['red','green','blue'], hue_order=['euclidean','cosine','correlation'], style='m', style_order=[3,2], ax=ax)
    g.set_xlim(30,200)
    g.set_ylim(-.1,.6)
    g.set_xlabel('Neighborhood Size [knn]')
    g.set_ylabel('Silhouette Index')
    g.legend(loc='lower left', ncol=2)
    si_vs_knn_plots[data_input] = fig
    plt.close()

pn.Row(si_vs_knn_plots['Original'], si_vs_knn_plots['Phase Randomization'], si_vs_knn_plots['Connection Randomization'])

# ***
# ## Panel A - Representative Embeddings
#
# Panel A also shows 2D embeddings for a few scenarios (combinations of knn and data input). The embeddings are the overlap of the embeddings for all scans after a procrustes transformation

dist='correlation'
m=2
accessory_plots = {}
for data_input in tqdm(['Original','Phase Randomization','Connection Randomization'], desc='Data Input'):
    for knn in [30,65,100,135,170,200]:
        # Grabs embeddings for the selected scenario
        sel_data           = si_xrs[data_input].loc[:,dist,knn,m].to_dataframe('SI').drop(['dist','knn','m'], axis=1)
        # Identify the best scan in this scenario to use as reference for subsequent procrustes transformations
        best_scan          = sel_data.idxmax()['SI']
        # Get a list with the names of all other scans that need to be transformed
        scans_to_transform = list(sel_data.index)
        scans_to_transform.remove(best_scan)
        # Copy the embedding to use as reference into the ref variable
        ref = embs_xrs[data_input][(best_scan,dist,knn,m)]
        # Create object that will contain all overlapped embeddings
        all_embs           = zscore(ref.copy()).reset_index()
        # Go one-by-one computing transformation and keeping it 
        for scan in scans_to_transform:
            aux            = embs_xrs[data_input][(scan,dist,knn,m)]
            _, aux_trf, m2 = procrustes(ref,aux)
            aux_trf        = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
            all_embs       = all_embs.append(aux_trf).reset_index(drop=True)
        # Create the plot
        accessory_plots[(data_input,knn,dist,m)] = all_embs.hvplot.scatter(x='LE001',y='LE002',c='Window Name', cmap=task_cmap, aspect='square', s=2, alpha=.3, legend=False, xaxis=False, yaxis=False).opts(toolbar=None, show_frame=False)

pn.GridBox(*[accessory_plots[k] for k in accessory_plots.keys()], ncols=6)

# ***
# # Panel B - Significant Differences in SI for best case scenario at 3D
#
# First, we find the knn that leads to the best SI for 3d embeddings

si_3d = si_xrs['Original'].loc[:,:,:,3].copy()
knn_max_si = int(si_3d.mean(dim='subject').idxmax(dim='knn').max().values)
print('++ INFO: knn leading to the maximum average SI across all subjects and metrics is: knn=%d' % knn_max_si)

# Next, we extract the SI values for all scans in that particular scenario

df2 = si_xrs['Original'].loc[:,:, knn_max_si,3].to_dataframe(name='SI').reset_index().drop(['knn','m'], axis=1)
df2.infer_objects()
df2.head(3)

# Finally, we generate the box plot and annotate it with statistical significance markers

fig, ax = plt.subplots(1,1,figsize=(4,7))
sns.set(font_scale=2)
g = sns.boxplot(data=df2, x='dist', y='SI', palette=['blue','green','red'], order=['correlation','cosine','euclidean'], ax=ax)
plt.xticks(rotation = 45)
pairs = [('euclidean','correlation'),('euclidean','cosine'),('correlation','cosine')]
annot = Annotator(g, pairs, data=df2, x='dist', y='SI', order=['correlation','cosine','euclidean'])
annot.configure(test='Wilcoxon', verbose=2, comparisons_correction='Bonferroni', text_format="star")
annot.apply_test()
annot.annotate()
ax.set_xlabel('Distance Function')
ax.set_ylabel('Silhouette Index')

# ***
# # Panel C - Embedding for all multi-task scans under the best scenario

dist = 'correlation'
knn = 65
m = 3
data_input = 'Original'

# +
sel_data           = si_xrs[data_input].loc[:,dist,knn,m].to_dataframe('SI').drop(['dist','knn','m'], axis=1)
best_scan          = sel_data.idxmax()['SI']
scans_to_transform = list(sel_data.index)
scans_to_transform.remove(best_scan)
print('++ Best scan to use as reference is: %s' % best_scan)
ref = embs_xrs[data_input][(best_scan,dist,knn,m)]
# Create Final Ouput Object
all_embs           = zscore(ref.copy()).reset_index()

for scan in scans_to_transform:
    aux            = embs_xrs[data_input][(scan,dist,knn,m)]
    _, aux_trf, m2 = procrustes(ref,aux)
    aux_trf        = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
    all_embs       = all_embs.append(aux_trf).reset_index(drop=True)
# -

camera = dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=1))
scene_correct_le = dict(
        xaxis = dict(nticks=1, gridcolor="rgb(230,230,230)", showbackground=True, zerolinecolor="white",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=1, gridcolor="rgb(230,230,230)", showbackground=True, zerolinecolor="white",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=1, gridcolor="rgb(230,230,230)", showbackground=True, zerolinecolor="white",backgroundcolor='rgb(230,230,230)'))

fig = px.scatter_3d(all_embs,x='LE001',y='LE002',z='LE003', 
              width=500, height=500, 
              opacity=0.3, color='Window Name',color_discrete_sequence=['gray','black','blue','yellow','green'])
fig.update_layout(showlegend=False, 
                          font_color='white', scene_aspectmode='cube');
fig.update_layout(scene=scene_correct_le, margin=dict(l=2, r=2, b=0, t=0, pad=0))
fig.update_traces(marker_size = 2)
fig

# ***
#
# # Preliminary Code to generate Supplementary Figure X

# Find case with maximum SI for the phase randomization model
si_xrs['Phase Randomization'].max()

# + tags=[]
si_xrs['Phase Randomization'].where(si_xrs['Phase Randomization']> .24).to_dataframe(name='SI').dropna().sort_values(by=['m','SI'], ascending=False)
# -

sbj = 'SBJ08'
dist = 'euclidean'
knn = 65

sbj = 'SBJ09'
dist = 'euclidean'
knn = 90

path = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data_Interim/PNAS2015/{sbj}/{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.pkl'.format(sbj=sbj)
swc = pd.read_pickle(path)

plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(30,5))
mat = sns.heatmap(swc,cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False,cbar_kws={'label': 'tvFC (Z)'})
mat.set_xlabel('Time [Windows]', fontsize=26)
mat.set_ylabel('Connections',    fontsize=26)
#for x,l in zip(tick_idxs,tick_labels):
#    mat.add_patch(Rectangle((x-45, -301),  91, 300, fill=True, color=task_cmap_caps[l], lw=0, clip_on=False))
cbar = mat.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.label.set_size(26)
cbar.ax.yaxis.set_label_position('left')

path = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data_Interim/PNAS2015/{sbj}/Null_ConnRand/{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.pkl'.format(sbj=sbj)
swc = pd.read_pickle(path)

plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(30,5))
mat = sns.heatmap(swc,cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False,cbar_kws={'label': 'tvFC (Z)'})
mat.set_xlabel('Time [Windows]', fontsize=26)
mat.set_ylabel('Connections',    fontsize=26)
#for x,l in zip(tick_idxs,tick_labels):
#    mat.add_patch(Rectangle((x-45, -301),  91, 300, fill=True, color=task_cmap_caps[l], lw=0, clip_on=False))
cbar = mat.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.label.set_size(26)
cbar.ax.yaxis.set_label_position('left')

path = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data_Interim/PNAS2015/{sbj}/Null_PhaseRand/{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.pkl'.format(sbj=sbj)
swc = pd.read_pickle(path)

plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(30,5))
mat = sns.heatmap(swc,cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False,cbar_kws={'label': 'tvFC (Z)'})
mat.set_xlabel('Time [Windows]', fontsize=26)
mat.set_ylabel('Connections',    fontsize=26)
#for x,l in zip(tick_idxs,tick_labels):
#    mat.add_patch(Rectangle((x-45, -301),  91, 300, fill=True, color=task_cmap_caps[l], lw=0, clip_on=False))
cbar = mat.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.label.set_size(26)
cbar.ax.yaxis.set_label_position('left')
