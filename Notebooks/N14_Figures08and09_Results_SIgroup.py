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

# # Description
#
# This notebook generates the different panels in Figures 8, 9 and Suppl. Fig 5, which summarizes the results of the SI evaluation at the group level.

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
import xarray as xr
import os.path as osp
import numpy as np
from statannotations.Annotator import Annotator

from utils.basics import PNAS2015_subject_list, PRJ_DIR, wls, wss
from pylab import subplot
from utils.io import load_LE_SI, load_TSNE_SI, load_UMAP_SI
# -

# Create Empty dictionary to hold all SI values
si       = {}

# # 1. Load Group Level SIs for all MLTs
#
# ## 1.1. Load SI for Laplacian Eigenmaps
# In this case, we restrict results to: Original Data & Correlation distance

# %%time
si['LE'] = load_LE_SI(sbj_list=['ALL','Procrustes'],check_availability=False, verbose=True, wls=wls, wss=wss,   ms=[2,3,5,10,15,20,25,30], dist_metrics=['correlation'], input_datas=['Original'])
si['LE'] = si['LE'].set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()

# ## 1.2. Load SI for UMAP
#
# In this case we restrict results to: Original Data & Euclidean distance

# %%time
si['UMAP'] = load_UMAP_SI(sbj_list=['ALL','Procrustes'],check_availability=False, verbose=True, wls=wls, wss=wss,   ms=[2,3,5,10,15,20,25,30], dist_metrics=['euclidean'], input_datas=['Original'])
si['UMAP'] = si['UMAP'].set_index(['Subject','Input Data','Norm','Metric','Knn','m','Alpha','Init','MinDist','Target']).sort_index()

# ## 1.3. Load SI for TSNE
#
# In this case, we restrict results to: Original Data & Correlation Distance & Alpha = 10,1000 & M = 2,3,5,10.
#
# This is necessary given how long it takes for T-SNE to run

# %%time
aux_TSNE_all        = load_TSNE_SI(sbj_list=['ALL'],check_availability=False,        verbose=True, wls=wls, wss=wss,   ms=[2,3,5,10], dist_metrics=['correlation'], input_datas=['Original'], alphas=[10,1000])
aux_TSNE_procrustes = load_TSNE_SI(sbj_list=['Procrustes'],check_availability=False, verbose=True, wls=wls, wss=wss,   ms=[2,3,5,10,15,20,25,30], dist_metrics=['correlation'], input_datas=['Original'], alphas=[10,1000])
si['TSNE']          = pd.concat([aux_TSNE_all,aux_TSNE_procrustes])
si['TSNE']          = si['TSNE'].set_index(['Subject','Input Data','Norm','Metric','PP','m','Alpha','Init','Target']).sort_index()

# ## 1.4. Summarize loaded data and change labels for plotting purposes

print('LE:   ' , [(name,len(si['LE'].index.get_level_values(name).unique())) for name in si['LE'].index.names])
print('TSNE: ' , [(name,len(si['TSNE'].index.get_level_values(name).unique())) for name in si['TSNE'].index.names])
print('UMAP: ' , [(name,len(si['UMAP'].index.get_level_values(name).unique())) for name in si['UMAP'].index.names])

for tech in ['LE','TSNE','UMAP']:
    index_names = list(si[tech].index.names)
    index_names = [name if name != 'Input Data' else 'Input' for name in index_names]
    index_names = [name if name != 'Metric'     else 'Distance' for name in index_names]
    index_names = [name if name != 'Norm'       else 'Normalization' for name in index_names]
    index_names = [name if name != 'Subject'    else 'Grouping Method' for name in index_names]

    
    si[tech].reset_index(inplace=True)
    si[tech].rename(columns={"Metric":"Distance","Input Data":"Input","Norm":"Normalization","Subject":"Grouping Method"},inplace=True)
    
    si[tech].replace('asis','None', inplace=True)
    si[tech].replace('zscored','Z-score', inplace=True)
    si[tech].replace('ALL','Concat. + '+tech, inplace=True)
    si[tech].replace('Procrustes',tech+' + Procrustes', inplace=True)
    si[tech].replace('correlation','Correlation', inplace=True)
    si[tech].replace('euclidean','Euclidean', inplace=True)
    si[tech].replace('cosine','Cosine', inplace=True)
    si[tech].replace('Null_ConnRand','Conn. Rand.', inplace=True)
    si[tech].replace('Null_PhaseRand','Phase Rand.', inplace=True)
    si[tech].set_index(index_names, inplace=True)

# + [markdown] tags=[]
# # 2. Results for Laplacian Eigenmaps
#
# * We will plot distributions of $SI_{task}$ and $SI_{subject}$ for both group aggregation methods.
# * For $SI_{task}$, we will aditionally highlight the portion of the distribution for the "LE + Procrustes" approach that relies on m > 3.
# -

si['LE'] = si['LE'].reset_index().replace({'LE + Procrustes':'Embed + Procrustes','Concat. + LE':'Concatenate + Embed'}).set_index(['Grouping Method','Input','Normalization','Distance','Knn','m','Target'])

# +
sns.set(font_scale=1.5, style='whitegrid')
fig,axs=plt.subplots(1,2,figsize=(20,5))
sns.histplot(data=si['LE'].loc[:,'Original',:,:,:,:,'Window Name'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes','Concatenate + Embed'], palette=['Orange','Blue'], kde=False, ax=axs[0], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.4)
sns.histplot(data=si['LE'].loc['Embed + Procrustes','Original',:,:,:,[5,10,15,20,25,30],'Window Name'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes'], palette=['DarkRed'], lw=3, kde=False, ax=axs[0], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.1, element='poly', legend=False)
axs[0].set_ylim(0,250);
axs[0].set_ylabel('Number of Times');
axs[0].set_xlabel('$SI_{task}$')
axs[0].set_xlim(-.2,.8)
axs[0].set_title('Group-Level LE Embeddings | Task Separability')

sns.histplot(data=si['LE'].loc[:,'Original',:,:,:,:,'Subject'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes','Concatenate + Embed'], palette=['Orange','Blue'], kde=False, ax=axs[1], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.4)
axs[1].set_ylim(0,250);
axs[1].set_ylabel('Number of Times');
axs[1].set_xlabel('$SI_{subject}$')
axs[1].set_xlim(-.2,.8)
axs[1].set_title('Group-level LE Embeddings | Subject Separability')
# -

# # 3. Results for UMAP
#
# * We will plot distributions of $SI_{task}$ and $SI_{subject}$ for both group aggregation methods.
# * For $SI_{task}$, we will aditionally highlight the portion of the distribution for the "UMAP + Procrustes" approach that relies on m > 3.
# * For $SI_{subject}$, we will highlight the portions of the "Concat + UMAP" distribution that corresponds to normalized and non-normalized data.

si['UMAP'] = si['UMAP'].reset_index().replace({'UMAP + Procrustes':'Embed + Procrustes','Concat. + UMAP':'Concatenate + Embed'}).set_index(['Grouping Method','Input','Normalization','Distance','Knn','m','Alpha','Init','MinDist','Target'])

sns.set(font_scale=1.5, style='whitegrid')
fig,axs=plt.subplots(1,2,figsize=(20,5))
sns.histplot(data=si['UMAP'].loc[:,'Original',:,:,:,:,:,:,:,'Window Name'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes','Concatenate + Embed'], palette=['Orange','Blue'], kde=False, ax=axs[0], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.4)
sns.histplot(data=si['UMAP'].loc['Embed + Procrustes','Original',:,:,:,[5,10,15,20,25,30],:,:,:,'Window Name'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes'], palette=['DarkRed'], kde=False, ax=axs[0], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.1, lw=3, element='poly', legend=False)
axs[0].set_ylim(0,250);
axs[0].set_ylabel('Number of Times');
axs[0].set_xlabel('$SI_{task}$')
axs[0].set_xlim(-.2,.8)
axs[0].set_title('Group-Level UMAP Embeddings | Task Separability')
sns.histplot(data=si['UMAP'].loc[:,'Original',:,:,:,:,:,:,:,'Subject'].reset_index(),      x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes','Concatenate + Embed'], palette=['Orange','Blue'], kde=False, ax=axs[1], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.4)
sns.histplot(data=si['UMAP'].loc['Concatenate + Embed','Original','None',:,:,:,:,:,:,'Subject'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Concatenate + Embed'], palette=['DarkBlue'], kde=False, ax=axs[1], lw=3, bins=np.linspace(start=-.2,stop=1,num=100), alpha=.3,element='poly', legend=False)
sns.histplot(data=si['UMAP'].loc['Concatenate + Embed','Original','Z-score',:,:,:,:,:,:,'Subject'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Concatenate + Embed'], palette=['Cyan'], kde=False, ax=axs[1], lw=3, bins=np.linspace(start=-.2,stop=1,num=100), alpha=.1,element='poly', legend=False)
axs[1].set_ylim(0,250);
axs[1].set_ylabel('Number of Times');
axs[1].set_xlabel('$SI_{subject}$')
axs[1].set_xlim(-.2,.8)
axs[1].set_title('Group-Level UMAP Embeddings | Subject Separability')

# We now find the best performing embeddings, as those will be added to the figure as representative results

data_wn = si['UMAP'].loc[:,:,:,:,:,:,:,:,:,'Window Name'].reset_index()
data_sb = si['UMAP'].loc[:,:,:,:,:,:,:,:,:,'Subject'].reset_index()

data_wn.sort_values(by='SI',ascending=False).iloc[0:2]

data_sb.sort_values(by='SI',ascending=False).iloc[0:2]

# # 4. Results for TSNE

si['TSNE'] = si['TSNE'].reset_index().replace({'TSNE + Procrustes':'Embed + Procrustes','Concat. + TSNE':'Concatenate + Embed'}).set_index(['Grouping Method','Input','Normalization','Distance','PP','m','Alpha','Init','Target'])

# +
sns.set(font_scale=1.5, style='whitegrid')
fig,axs=plt.subplots(1,2,figsize=(20,5))
sns.histplot(data=si['TSNE'].loc[:,'Original',:,:,:,:,:,:,'Window Name'].reset_index(),                            x='SI', hue='Grouping Method', hue_order=['Embed + Procrustes','Concatenate + Embed'], palette=['Orange','Blue'], kde=False, ax=axs[0], bins=30, alpha=.4)
sns.histplot(data=si['TSNE'].loc['Embed + Procrustes','Original',:,:,:,[5,10,15,20,25,30],:,:,'Window Name'].reset_index(), x='SI', hue='Grouping Method', hue_order=['Embed + Procrustes'], palette=['DarkOrange'], kde=False, ax=axs[0], bins=75, alpha=.3, lw=3, element='poly', legend=False)

axs[0].set_ylim(0,50);
axs[0].set_ylabel('Number of Times');
axs[0].set_xlabel('$SI_{task}$')
axs[0].set_xlim(-.2,.8)
axs[0].set_title('Group T-SN | Task Separability')
sns.histplot(data=si['TSNE'].loc[:,'Original',:,:,:,:,:,:,'Subject'].reset_index(), x='SI',hue='Grouping Method', hue_order=['Embed + Procrustes','Concatenate + Embed'], palette=['Orange','Blue'],      kde=False, ax=axs[1], bins=np.linspace(start=-.2,stop=1,num=100), alpha=.4)
sns.histplot(data=si['TSNE'].loc['Concatenate + Embed','Original','None',   :,:,:,:,:,'Subject'].reset_index(), x='SI',hue='Grouping Method',    hue_order=['Concatenate + Embed'], palette=['DarkBlue'], kde=False, ax=axs[1], lw=3, bins=np.linspace(start=-.2,stop=1,num=100), alpha=.3,element='poly', legend=False)
sns.histplot(data=si['TSNE'].loc['Concatenate + Embed','Original','Z-score',:,:,:,:,:,'Subject'].reset_index(), x='SI',hue='Grouping Method',    hue_order=['Concatenate + Embed'], palette=['Cyan'],     kde=False, ax=axs[1], lw=3, bins=np.linspace(start=-.2,stop=1,num=100), alpha=.1,element='poly', legend=False)
axs[1].set_ylim(0,50);
axs[1].set_ylabel('Number of Times');
axs[1].set_xlabel('$SI_{subject}$')
axs[1].set_xlim(-.2,.8)
axs[1].set_title('Group UMAP | Task Separability')
# -

data_wn = si['TSNE'].loc[:,:,:,:,:,:,:,:,'Window Name'].reset_index()
data_sb = si['TSNE'].loc[:,:,:,:,:,:,:,:,'Subject'].reset_index()

data_wn.sort_values(by='SI',ascending=False).iloc[0:2]

data_sb.sort_values(by='SI',ascending=False).iloc[0:2]
