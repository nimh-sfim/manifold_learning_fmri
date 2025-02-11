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

import panel as pn
import numpy as np
import os.path as osp
import pandas as pd
from scipy.stats import zscore
import hvplot.pandas
import plotly.express as px

# +
# #cd /data/SFIMJGC_HCP7T/manifold_learning_fmri/Notebooks
#panel convert GUI_Embeddings_toDeploy.py --to pyodide-worker --out ../docs/ --pwa --title manifold_fmri
# -

# So far we are working with these values of wls and wss across the whole manuscript
wls = 45
wss = 1.5
min_dist = 0.8

# +
DATA_URL = 'https://raw.githubusercontent.com/nimh-sfim/manifold_learning_fmri_demo_data/master/data/'

# Available scans
avail_scans_dict = {'Scan 1':'SBJ06', 'Scan 2':'SBJ07'}

# Available Data Scenarios
input_data_dict = {'Real Data':'Original','Connectivity Randomization':'Null_ConnRand','Phase Randomization':'Null_PhaseRand'}

# Normalization Options
normalization_dict = {'Do not normalize':'asis','Z-score':'zscored'}

# Colormaps
# sbj_cmap_list = [rgb2hex(c) for c in matplotlib.colormaps['tab20'].colors]
# Hard coded below to avoid importing matplotlib
sbj_cmap_list = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5']
sbj_cmap = {v:sbj_cmap_list[i] for i,v in enumerate(avail_scans_dict.values())}
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Vis. Motion':  '#F4D03F',  'Matemathics': 'green', 'Mixed Tasks': 'pink'}

# Laplacian Eigenmap related options
le_dist_metrics = {'Euclidean Distance':'euclidean','Correlation Distance':'correlation','Cosine Distance':'cosine'}
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
tsne_pps          = [5,50,100,150]
tsne_ms           = [2,3,5,10]
tsne_alphas       = [10, 50, 100, 1000]
tsne_inits        = ['pca']

# Camera configuration for 3D plots
camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=1, z=1))

# Task labels
new_label_dict = {'REST':'Rest','VIDE':'Vis. Motion','BACK':'Memory','MATH':'Matemathics','XXXX':'Mixed Tasks'}


# +
scat_2d_width   = 250
scat_2d_height  = 250
scat_2d_resize  = None 
scat_3d_width   = 350
scat_3d_height  = 300
scat_3d_resize  = None

sidebar_widgets_width = 200
sidebar_width = 250

tabs_widget_width = 150


# -

# ***
# ### Functions from utils.plotting

def plot_2d_scatter(data,x,y,c,cmap=task_cmap, show_frame=True, s=2, alpha=0.3, toolbar=None, 
                    legend=True, xaxis=False, xlabel='', yaxis=False, ylabel='', frame_width=scat_2d_width, shared_axes=False):
    plot = data.hvplot.scatter(x=x,y=y,c=c, cmap=cmap, 
                               aspect='square', s=s, alpha=alpha, 
                               legend=legend, xaxis=xaxis, 
                               yaxis=yaxis, frame_width=frame_width, shared_axes=shared_axes).opts(toolbar=toolbar, show_frame=show_frame, tools=[], legend_position='left')
    return plot


def plot_3d_scatter(data,x,y,z,c,cmap,s=2,width=scat_3d_width, height=scat_3d_height, ax_range=[-.005,.005],nticks=4):
    fig = px.scatter_3d(data,
                        x=x,y=y,z=z, 
                        width=width, height=height, 
                        opacity=0.3, color=c,color_discrete_sequence=cmap)
    scene_extra_confs = dict(
        xaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))
    fig.update_traces(marker_size = s)
    fig.update_layout(showlegend=False, font_color='white',title = dict(text="TITLE"),
                      scene_camera=camera, scene=scene_extra_confs, margin=dict(l=0, r=0, b=0, t=0))
    return fig


# ***
# ### Functions from utils.io

def load_single_le(sbj,input_data,scenario,dist,knn,m,wls=45,wss=1.5, drop_xxxx=True, show_path=False):
    path = osp.join(DATA_URL,'embeddings',sbj,'LE',input_data,
                    '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(sbj=sbj,scenario=scenario,wls=str(int(wls)).zfill(3),wss=str(wss),                                                                                                    dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4)))
    try:
        aux = pd.read_pickle(path)
    except:
        return None
    if drop_xxxx:
        if type(aux.index) is pd.MultiIndex:
            aux = aux.drop('XXXX', level='Window Name')
        else:
            aux = aux.drop('XXXX',axis=0)
    aux.index = [new_label_dict[i] for i in aux.index]
    aux.index.name = 'Task'
    return aux


def load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,m,wls=45,wss=1.5, drop_xxxx=True):
    path = osp.join(DATA_URL,'embeddings',sbj,'TSNE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,
                                                                                                init_method=init_method,sbj=sbj,wls=str(int(wls)).zfill(3),wss=str(wss),
                                                                                                dist=dist,pp=str(pp).zfill(4),m=str(m).zfill(4),alpha=str(alpha)))
    try:
        aux = pd.read_pickle(path)
    except:
        return None
    if drop_xxxx:
        if type(aux.index) is pd.MultiIndex:
            aux = aux.drop('XXXX', level='Window Name')
        else:
            aux = aux.drop('XXXX',axis=0)
    aux.index = [new_label_dict[i] for i in aux.index]
    aux.index.name = 'Task'
    return aux


def load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,m,wls=45,wss=1.5, drop_xxxx=True):
    path = osp.join(DATA_URL,'embeddings',sbj,'UMAP',input_data,
                    '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,        
                                                                                                init_method=init_method,sbj=sbj,wls=str(int(wls)).zfill(3),wss=str(wss),
                                                                                                dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4),min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
    try:
        aux = pd.read_pickle(path)
    except:
        return None
    if drop_xxxx:
        if type(aux.index) is pd.MultiIndex:
            aux = aux.drop('XXXX', level='Window Name')
        else:
            aux = aux.drop('XXXX',axis=0)
    aux.index = [new_label_dict[i] for i in aux.index]
    aux.index.name = 'Task'
    return aux


# ***
# # Main Dashboard Panel: Configuration Options

sidebar_desc = pn.pane.Markdown('#### Use these widgets to select input data entering the embedding estimation', width=sidebar_widgets_width)
sbj_select      = pn.widgets.Select(name='fMRI Scan',     options=avail_scans_dict,   width=sidebar_widgets_width, description='Select the scan you want to explore')
input_select    = pn.widgets.Select(name='Scenario',      options=input_data_dict,    width=sidebar_widgets_width, description='Select original data or null data (phase or connection randomized)')
scenario_select = pn.widgets.Select(name='Normalization', options=normalization_dict, width=sidebar_widgets_width,description='Select whether or not to normalize data prior to embedding estimation')
sidebar_divider = pn.layout.Divider()
sidebar_todo    = pn.pane.Markdown("""
#### Things you can do:
1. Check how distance and neighborhood size affect embedding quality.
2. Get a feeling for inter-subject variability by comparing results for different scans.
3. See how randomizing connections or the phase of timeseries removes task structure from embeddings.
4. Explore differences across Manifold Learning methods when keeping the input data unchanged.
""",
width=sidebar_widgets_width)

# ***
# # Laplacian Eigenmaps

# #### 1. Load Silhouette Index for LE

si_LE_URL = osp.join(DATA_URL,'sil_index','si_LE.pkl')
si_LE = pd.read_pickle(si_LE_URL)

# #### 3. LE Tab Elements

CSS = """
input { height: 15px; width: 15px;}
span { font-size: 16px; }
"""
le_m_select     = pn.widgets.Select(name='M',   options=le_ms, value=le_ms[-1], width=tabs_widget_width, description='Number of dimensions used for computing the left-most embedding (independently of M, the plot will only show the first three dimensions)')
le_knn_select   = pn.widgets.Select(name='Knn', options=le_knns,         value=le_knns[0], width=tabs_widget_width, description='Neighborhood Size for Laplacian Embeddings')
le_dist_select  = pn.widgets.Select(name='Distance Metric', options=le_dist_metrics, width=tabs_widget_width,description='Distance metric used when computing Laplacian Embeddings')
le_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows?', width=tabs_widget_width, align=('center','center'), margin=(20,15),stylesheets=[CSS])
le_conf_box     = pn.Row(le_dist_select,le_knn_select,le_m_select,le_drop_xxxx)


def plot_LE_scats(group_type,input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
   # Load all necessary embeddings
    # =============================
    if m == 2:
        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_le(group_type,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_le(group_type,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_le(group_type,input_data,scenario,dist,knn,m,drop_xxxx=drop_xxxx)
    # Preprare Embeddings
    # ===================
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        
    if not (aux_3d is None):
         aux_3d = aux_3d.apply(zscore)
         aux_3d = aux_3d.reset_index()

    if not (aux_Md is None):
         aux_Md = aux_Md.apply(zscore)
         aux_Md = aux_Md.reset_index()
    # Prepare Color-scales
    # ====================
    if color_col == 'Subject':
        cmap_2d = sbj_cmap
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Task'].unique()]

    # Plotting
    # ========
    if (not (aux_2d is None)) & (aux_3d is None):
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_LE.loc[group_type,input_data,scenario,dist,knn,2,'Task']['SI'].item())
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        plots = pn.Row(pn.Column(col_title_2d,emb_plot_2d),None,None)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_LE.loc[group_type,input_data,scenario,dist,knn,2,'Task']['SI'].item())
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        col_title_3d = pn.pane.Markdown("## 3D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_LE.loc[group_type,input_data,scenario,dist,knn,3,'Task']['SI'].item())
        emb_plot_3d  = plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_3d.min(),aux_3d.max()])
        plots = pn.Row(pn.Column(col_title_2d,emb_plot_2d), pn.Column(col_title_3d,emb_plot_3d),None)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_LE.loc[group_type,input_data,scenario,dist,knn,2,'Task']['SI'].item())
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        col_title_3d = pn.pane.Markdown("## 3D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_LE.loc[group_type,input_data,scenario,dist,knn,3,'Task']['SI'].item())
        emb_plot_3d  = plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_3d.min(),aux_3d.max()])
        col_title_Md = pn.pane.Markdown("## 3D View of %d-D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % (m,si_LE.loc[group_type,input_data,scenario,dist,knn,m,'Task']['SI'].item()))
        emb_plot_Md  = plot_3d_scatter(aux_Md,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_Md.min(),aux_Md.max()])
        plots = pn.GridBox(*[col_title_2d,col_title_3d,col_title_Md,
                             emb_plot_2d,emb_plot_3d,emb_plot_Md],ncols=3)
    return plots


@pn.depends(sbj_select,input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,le_drop_xxxx)
def plot_LE_Scan_scats(sbj,input_data,scenario,dist,knn,m, drop_xxxx):
    return plot_LE_scats(sbj,input_data,scenario,dist,knn,m,'Task','above',drop_xxxx)


le_config_card                = pn.Row(le_conf_box)
le_embs_scan_card             = pn.layout.Card(plot_LE_Scan_scats,title='Laplacian Eigenmaps - Single fMRI Scan', header_background='#0072B5', header_color='#ffffff')
le_embs_col = pn.Column(le_embs_scan_card)

le_tab=pn.Column(le_config_card,le_embs_col)

# ***
# # UMAP
# #### 1. Load Silhouette Index for UMAP
#
# Hyper-parameter space: 3 Inputs * 2 Norm Approach * 8 m * 3 dist * x knns * 3 alphas = 
# * "Concat + UMAP": 17280 entries
# * "UMAP + Procrustes": 17280 entries
# * Single-Scan Level: 345600 entries

si_UMAP = pd.read_pickle(osp.join(DATA_URL,'sil_index','si_UMAP.pkl'))

# #### 3. UMAP Tab Elements

# +
umap_knn_select   = pn.widgets.Select(name='Knn',             options=umap_knns,         value=umap_knns[0], width=tabs_widget_width, description='Select a neighborhood size for UMAP.')
umap_dist_select  = pn.widgets.Select(name='Distance Metric', options=umap_dist_metrics, width=tabs_widget_width, description='Select a distance metric for UMAP.')
umap_m_select     = pn.widgets.Select(name='M',   options=umap_ms,           value=umap_ms[0], width=tabs_widget_width, description='Select the number of dimensions for UMAP.')
umap_alpha_select = pn.widgets.Select(name='Learning Rate',   options=umap_alphas,       value=umap_alphas[0], width=tabs_widget_width, description='Select a learning rate for UMAP.')
umap_init_select  = pn.widgets.Select(name='Init Method',     options=['spectral'],        value='spectral', width=tabs_widget_width, description='Initialization method set to spectral.')
umap_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows?', width=tabs_widget_width, align=('center','center'), margin=(20,15),stylesheets=[CSS])

umap_conf_box     = pn.Row(umap_dist_select,umap_knn_select,umap_init_select,umap_m_select,umap_alpha_select,umap_drop_xxxx)
umap_LEFT_col     = pn.Column(umap_conf_box)


# -

# #### 3. Plotting Functions            

def plot_UMAP_scats(group_type,input_data,scenario,dist,knn,m,alpha,init_method,min_dist,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    # =============================
    if m == 2:
        aux_2d = load_single_umap(group_type,input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_umap(group_type,input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap(group_type,input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_umap(group_type,input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap(group_type,input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_umap(group_type,input_data,scenario,dist,knn,alpha,init_method,min_dist,m,drop_xxxx=drop_xxxx)
    # Preprare Embeddings
    # ===================
    if not (aux_2d is None):
         aux_2d = aux_2d.apply(zscore)
         aux_2d = aux_2d.reset_index()
    if not (aux_3d is None):
         aux_3d = aux_3d.apply(zscore)
         aux_3d = aux_3d.reset_index()
    if not (aux_Md is None):
         aux_Md = aux_Md.apply(zscore)
         aux_Md = aux_Md.reset_index()
    # Prepare Color-scales
    # ====================
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Task'].unique()]
    # Plotting
    # ========
    if (not (aux_2d is None)) & (aux_3d is None):
        si_UMAP_2d   = si_UMAP.loc[group_type,input_data,scenario,init_method,min_dist,dist,knn,alpha,2,'Task'].round(2).item()
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_UMAP_2d)
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        plots = pn.Row(pn.Column(col_title_2d,emb_plot_2d),None,None)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        si_UMAP_2d   = si_UMAP.loc[group_type,input_data,scenario,init_method,min_dist,dist,knn,alpha,2,'Task'].round(2).item()
        si_UMAP_3d   = si_UMAP.loc[group_type,input_data,scenario,init_method,min_dist,dist,knn,alpha,3,'Task'].round(2).item()
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_UMAP_2d)
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        col_title_3d = pn.pane.Markdown("## 3D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_UMAP_3d)
        emb_plot_3d  = plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_3d.min(),aux_3d.max()])
        plots = pn.Row(pn.Column(col_title_2d,emb_plot_2d), pn.Column(col_title_3d,emb_plot_3d),None)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        si_UMAP_2d   = si_UMAP.loc[group_type,input_data,scenario,init_method,min_dist,dist,knn,alpha,2,'Task'].round(2).item()
        si_UMAP_3d   = si_UMAP.loc[group_type,input_data,scenario,init_method,min_dist,dist,knn,alpha,3,'Task'].round(2).item()
        si_UMAP_Md   = si_UMAP.loc[group_type,input_data,scenario,init_method,min_dist,dist,knn,alpha,m,'Task'].round(2).item()
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_UMAP_2d)
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        col_title_3d = pn.pane.Markdown("## 3D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_UMAP_3d)
        emb_plot_3d  = plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_3d.min(),aux_3d.max()])
        col_title_Md = pn.pane.Markdown("## 3D View of %d-D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % (m,si_UMAP_Md))
        emb_plot_Md  = plot_3d_scatter(aux_Md,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_Md.min(),aux_Md.max()])
        plots = pn.GridBox(*[col_title_2d,col_title_3d,col_title_Md,
                             emb_plot_2d,emb_plot_3d,emb_plot_Md],ncols=3)
    return plots


@pn.depends(sbj_select,input_select,scenario_select,umap_dist_select,umap_knn_select,umap_m_select,umap_alpha_select,umap_init_select,umap_drop_xxxx)
def plot_UMAP_Scan_scats(sbj,input_data,scenario,dist,knn,m,alpha,init_method,drop_xxxx):
    return plot_UMAP_scats(sbj,input_data,scenario,dist,knn,m,alpha,init_method,0.8,'Task','above',drop_xxxx)


# #### 4. Constructing UMAP Tab with all elements

umap_embs_scan_card = pn.layout.Card(plot_UMAP_Scan_scats,title='UMAP Embeddings - Single fMRI Scan', header_background='#0072B5', header_color='#ffffff')
umap_embs_col       = pn.Column(umap_embs_scan_card)

umap_tab = pn.Column(umap_LEFT_col,umap_embs_col)

# ***
# # TSNE
# #### 1. Load Silhouette Index for TSNE

si_TSNE_URL = osp.join(DATA_URL,'sil_index','si_TSNE.pkl')
si_TSNE = pd.read_pickle(si_TSNE_URL)

# #### 2. TSNE Tab Elements

# +
tsne_pp_select   = pn.widgets.Select(name='Perplexity',        options=tsne_pps,          value=50, width=tabs_widget_width, description='Choose the desired perplexity, which is similar to the neighborhood size.')
tsne_dist_select  = pn.widgets.Select(name='Distance Metric',  options=tsne_dist_metrics, value='correlation', width=tabs_widget_width, description='Choose a distance metric.')
tsne_m_select     = pn.widgets.Select(name='M',                options=tsne_ms,       value=2, width=tabs_widget_width, description='Number of dimensions used during embedding estimation')
tsne_alpha_select = pn.widgets.Select(name='Learning Rate',    options=tsne_alphas,       value=tsne_alphas[0], width=tabs_widget_width, description='Choose the learning rate')
tsne_init_select  = pn.widgets.Select(name='Init Method',      options=tsne_inits,       value=tsne_inits[0], width=tabs_widget_width, description='We always initialize using the PCA method')
tsne_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows?', width=tabs_widget_width, align=('center','center'), margin=(20,15),stylesheets=[CSS])

tsne_conf_box     = pn.Row(tsne_dist_select,tsne_pp_select,tsne_init_select,tsne_m_select,tsne_alpha_select,tsne_drop_xxxx)
tsne_LEFT_col     = pn.Row(tsne_conf_box)


# -

# #### 3. Plotting Functions

def plot_TSNE_scats(group_type,input_data,scenario,dist,pp,m,alpha,init_method,color_col,plot_2d_toolbar, drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    # =============================
    if m == 2:
        aux_2d = load_single_tsne(group_type,input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_tsne(group_type,input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_tsne(group_type,input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_tsne(group_type,input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_tsne(group_type,input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_tsne(group_type,input_data,scenario,dist,pp,alpha,init_method,m,drop_xxxx=drop_xxxx)
    # Preprare Embeddings for plotting purposes
    # =========================================
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
    # Prepare Color-scales
    # ====================
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Task'].unique()]    
     # Plotting
    # ========
    if (not (aux_2d is None)) & (aux_3d is None):
        si_tsne_2d   = si_TSNE.loc[group_type,input_data,scenario,dist,pp,2,alpha,init_method,'Task'].round(2).item()
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_tsne_2d)
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        plots = pn.Row(pn.Column(col_title_2d,emb_plot_2d),None,None)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        si_tsne_2d   = si_TSNE.loc[group_type,input_data,scenario,dist,pp,2,alpha,init_method,'Task'].round(2).item()
        si_tsne_3d   = si_TSNE.loc[group_type,input_data,scenario,dist,pp,3,alpha,init_method,'Task'].round(2).item()
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_tsne_2d)
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        col_title_3d = pn.pane.Markdown("## 3D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_tsne_3d)
        emb_plot_3d  = plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_3d.min(),aux_3d.max()])
        plots = pn.Row(pn.Column(col_title_2d,emb_plot_2d), pn.Column(col_title_3d,emb_plot_3d),None)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        si_tsne_2d   = si_TSNE.loc[group_type,input_data,scenario,dist,pp,2,alpha,init_method,'Task'].round(2).item()
        si_tsne_3d   = si_TSNE.loc[group_type,input_data,scenario,dist,pp,3,alpha,init_method,'Task'].round(2).item()
        si_tsne_Md   = si_TSNE.loc[group_type,input_data,scenario,dist,pp,m,alpha,init_method,'Task'].round(2).item()
        col_title_2d = pn.pane.Markdown("## 2D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_tsne_2d)
        emb_plot_2d  = plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar)
        col_title_3d = pn.pane.Markdown("## 3D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % si_tsne_3d)
        emb_plot_3d  = plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_3d.min(),aux_3d.max()])
        col_title_Md = pn.pane.Markdown("## 3D View of %d-D Embedding | <a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>SI</a>=%.2f" % (m,si_tsne_Md))
        emb_plot_Md  = plot_3d_scatter(aux_Md,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[aux_Md.min(),aux_Md.max()])
        plots = pn.GridBox(*[col_title_2d,col_title_3d,col_title_Md,
                             emb_plot_2d,emb_plot_3d,emb_plot_Md],ncols=3)
    return plots


@pn.depends(sbj_select,input_select,scenario_select,tsne_dist_select,tsne_pp_select,tsne_m_select,tsne_alpha_select,tsne_init_select,tsne_drop_xxxx)
def plot_TSNE_Scan_scats(sbj,input_data,scenario,dist,pp,m,alpha,init_method,drop_xxxx):
    return plot_TSNE_scats(sbj,input_data,scenario,dist,pp,m,alpha,init_method,'Task','above', drop_xxxx)


# #### 4. Put the T-SNE Tab elements together

tsne_embs_scan_card             = pn.layout.Card(plot_TSNE_Scan_scats,title='T-SNE Embeddings - Single fMRI Scan', header_background='#0072B5', header_color='#ffffff')
tsne_embs_col                   = pn.Column(tsne_embs_scan_card)

tsne_tab = pn.Column(tsne_LEFT_col,tsne_embs_col)

# ***

# +
config = {"headerControls": {"close": "remove","maximize":"remove"}}

intro_img  = pn.pane.Image("https://raw.githubusercontent.com/nimh-sfim/manifold_learning_fmri/master/docs/images/Embedding_GUI_Intro.png", width=480, align=('center','center'))
intro_text = pn.pane.Markdown("""
This dashbaord allows you to explore time-vayring fMRI data embedded using [T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), [UMAP](https://umap-learn.readthedocs.io/en/latest/) and [Laplacian Eigenamps](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html). 

All details about this work can be found in [our scientific publication](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1134012/full) in the journal [Frontiers in Human Neuroscience](https://www.frontiersin.org/journals/human-neuroscience).

In a nutshell, fMRI data was acquired for 25 minutes as subjects performed four different cognitive tasks (visual attention, 2-back working memory, simple math and rest). Using the [Craddock atlas](https://onlinelibrary.wiley.com/doi/10.1002/hbm.21333), and a [sliding window](https://www.sciencedirect.com/science/article/pii/S1053811919300874?via%3Dihub) apporach we generated whole-brain, time-varying functional connectivity matrices (left). These are high dimensional matrices (edges X time); and therefore hard to interpret. To evaluate if Manifold Learning can aid with interpretation, we constructed embeddings and checked how well they capture the task structue of the experiment.

Use, the widgets on the right sidebar to select input data. Use the widgets in each of the tabs below to see how technique specific hyperparameters affect the quality of the embedding and their ability to show the task structure. Have fun!!!!
""", width=650)

intro_frame = pn.layout.FloatPanel(pn.Row(intro_img, intro_text),name='Introduction and basic instructions', position='right-top', config=config)
# -

# Instantiate the template with widgets displayed in the sidebar
template = pn.template.FastListTemplate(
    title="Manifold Learning for time-varying functional connectivity",
    sidebar=[sidebar_desc,sbj_select,input_select,scenario_select, sidebar_divider,sidebar_todo, pn.layout.Divider()],
    sidebar_width=sidebar_width,
    theme_toggle=False,
)

spacer_for_intro_bar_when_minimized = pn.Spacer(styles=dict(background='#f7f7f7'),sizing_mode='stretch_both')
embedding_tabs                      = pn.Tabs(('Laplacian Eigenmaps',le_tab),('T-SNE',tsne_tab),('UMAP',umap_tab), sizing_mode='stretch_width')
template.main.append(pn.Column(intro_frame,
                               pn.Row(spacer_for_intro_bar_when_minimized, height=30),
                               embedding_tabs)) 

template.servable()

# +
# import os
# port_tunnel = int(os.environ['PORT2'])
# print('++ INFO: Second Port available: %d' % port_tunnel)
# dashboard = template.show(port=port_tunnel)
# -




