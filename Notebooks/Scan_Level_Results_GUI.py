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

# +
import panel as pn
import numpy as np
import os.path as osp
import pandas as pd
import os
from tqdm.notebook import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
from statannotations.Annotator import Annotator
from utils.plotting import generate_Avg_LE_SIvsKnn_ScanLevel,  generate_LE_SIvsKnn_ScanLevel,  generate_LE_SIvsKNN_GroupLevel
from utils.plotting import generate_Avg_UMAP_SIvsKnn_ScanLevel,generate_UMAP_SIvsKnn_ScanLevel,generate_UMAP_SIvsKNN_GroupLevel
from utils.plotting import generate_Avg_TSNE_SIvsKnn_ScanLevel,generate_TSNE_SIvsKnn_ScanLevel,generate_TSNE_SIvsKNN_GroupLevel

from sklearn.metrics import silhouette_score
from scipy.stats import zscore
# -

port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# +
from utils.basics import PRJ_DIR, PNAS2015_subject_list, sbj_cmap_list,sbj_cmap_dict
from utils.basics import le_dist_metrics, le_knns, le_ms
from utils.basics import umap_dist_metrics, umap_knns, umap_ms, umap_alphas, umap_inits
from utils.basics import tsne_dist_metrics, tsne_pps, tsne_ms, tsne_alphas, tsne_inits
from utils.basics import input_datas, norm_methods
from utils.basics import task_cmap_caps as task_cmap
from utils.basics import sbj_cmap_dict, sbj_cmap_list
from utils.plotting import plot_2d_scatter, plot_3d_scatter
from utils.io import load_LE_SI, load_UMAP_SI, load_TSNE_SI

from utils.plotting import get_SIvsKNN_plots
from utils.io import load_single_le, load_single_umap, load_single_tsne
#from utils.procrustes import procrustes_scan_embs
# -

# So far we are working with these values of wls and wss across the whole manuscript
wls = 45
wss = 1.5
min_dist = 0.8

# + [markdown] tags=[]
# ***
# # Main Dashboard Panel: Configuration Options
# -

sbj_select            = pn.widgets.Select(name='Subject',                 options=PNAS2015_subject_list, value=PNAS2015_subject_list[0], width=150)
input_select          = pn.widgets.Select(name='Input Data',              options=['Original','Null_ConnRand','Null_PhaseRand'],            value='Original', width=150)
scenario_select       = pn.widgets.Select(name='Normalization',           options=['asis','zscored'], value='asis', width=150)
plot2d_toolbar_select = pn.widgets.Select(name='2D Toolbar', options=['above', 'below', 'left', 'right', 'disable'], value='disable', width=150) 
data_select_box       = pn.Row(sbj_select,input_select, scenario_select, plot2d_toolbar_select, background='WhiteSmoke')
data_select_box

# ***
# # Laplacian Eigenmaps

# + [markdown] tags=[]
# #### 1. Load Silhouette Index for LE
# -

# %%time
RELOAD_SI_LE = False
if RELOAD_SI_LE:
    si_LE_all        = load_LE_SI(sbj_list=['ALL'],check_availability=False,               verbose=True, wls=wls, wss=wss,                ms=[2,3,5,10,15,20,25,30])
    si_LE_procrustes = load_LE_SI(sbj_list=['Procrustes'],check_availability=False,        verbose=True, wls=wls, wss=wss,         ms=[2,3,5,10,15,20,25,30])
    si_LE_scans      = load_LE_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3,5,10,15,20,25,30])
    si_LE            = pd.concat([si_LE_scans, si_LE_all, si_LE_procrustes])
    si_LE.replace('Window Name','Task', inplace=True)
    si_LE            = si_LE.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()
    si_LE.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_LE.pkl'))
    del si_LE_scans, si_LE_all, si_LE_procrustes
else:
    si_LE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_LE.pkl'))

# Change Normalization method labels so they agree with the text
si_LE = si_LE.reset_index().replace('asis','None')
si_LE = si_LE.replace('zscored','Z-score')
si_LE = si_LE.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()
si_LE.head()

# + [markdown] tags=[]
# #### 2. Generate Summary View Figues
# -

# %%time
REGENERATE_LE_KNN_PLOTS = False
if REGENERATE_LE_KNN_PLOTS:
    generate_Avg_LE_SIvsKnn_ScanLevel(si_LE,PNAS2015_subject_list)
    generate_LE_SIvsKNN_GroupLevel(si_LE)
    for sbj in PNAS2015_subject_list:
        generate_LE_SIvsKnn_ScanLevel(si_LE,sbj)

# Change Normalization method labels so they agree with the text
si_LE = si_LE.reset_index().replace('None','asis')
si_LE = si_LE.replace('Z-score','zscored')
si_LE = si_LE.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()
si_LE.head()

# #### 3. LE Tab Elements

le_m_select     = pn.widgets.Select(name='M',               options=[2,3,5,10,15,20,25,30],         value=5, width=150)
le_knn_select   = pn.widgets.Select(name='Knn',             options=le_knns,         value=le_knns[0], width=150)
le_dist_select  = pn.widgets.Select(name='Distance Metric', options=le_dist_metrics, value=le_dist_metrics[0], width=150)
le_grcc_col_sel = pn.widgets.Select(name='[G-CC] Color By:', options=['Window Name','Subject'], value='Window Name', width=150)
le_grpt_col_sel = pn.widgets.Select(name='[G-PT] Color By:', options=['Window Name','Subject'], value='Window Name', width=150)
le_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows', width=150)
le_conf_box     = pn.WidgetBox(le_dist_select,le_knn_select,le_m_select,le_grcc_col_sel,le_grpt_col_sel,le_drop_xxxx)
le_conf_box


# +
@pn.depends(sbj_select,input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,plot2d_toolbar_select,le_drop_xxxx)
def plot_LE_Scan_scats(sbj,input_data,scenario,dist,knn,m,plot_2d_toolbar, drop_xxxx):
    plots = None
    aux_2d = load_single_le(sbj,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)
    aux_3d = load_single_le(sbj,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)
    aux_Md = load_single_le(sbj,input_data,scenario,dist,knn,m,drop_xxxx=drop_xxxx)
    if not (aux_2d is None):
        aux_2d = aux_2d.reset_index()
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d_task_cmap = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (m<=3):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_LE.loc[sbj,input_data,scenario,dist,knn,2].round(2),width=150),
                             pn.pane.DataFrame(si_LE.loc[sbj,input_data,scenario,dist,knn,3].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (m>3):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='LE001',y='LE002',z='LE003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_LE.loc[sbj,input_data,scenario,dist,knn,2].round(2),width=150),
                             pn.pane.DataFrame(si_LE.loc[sbj,input_data,scenario,dist,knn,3].round(2),width=150),
                             pn.pane.DataFrame(si_LE.loc[sbj,input_data,scenario,dist,knn,m].round(2),width=150)],ncols=3)
    return plots

@pn.depends(input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,le_grcc_col_sel,plot2d_toolbar_select,le_drop_xxxx)
def plot_LE_Group_Concat_scats(input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d = load_single_le('ALL',input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)
    aux_3d = load_single_le('ALL',input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        cmap_3d = [task_cmap[t] for t in aux_3d.index.get_level_values('Window Name').unique()]
    if not (aux_2d is None):
        aux_2d = aux_2d.reset_index()
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d_task_cmap = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    if (not (aux_2d is None)) & (not (aux_3d is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_LE.loc['ALL',input_data,scenario,dist,knn,2].round(2),width=150),
                             pn.pane.DataFrame(si_LE.loc['ALL',input_data,scenario,dist,knn,2].round(3),width=150)],ncols=2)
    return plots

@pn.depends(input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,le_grpt_col_sel,plot2d_toolbar_select,le_drop_xxxx)
def plot_LE_Group_Procrustes_scats(input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d = load_single_le('Procrustes',input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)
    aux_3d = load_single_le('Procrustes',input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)
    aux_Md = None
    if m > 3:
        aux_Md = load_single_le('Procrustes',input_data,scenario,dist,knn,m,drop_xxxx=drop_xxxx)
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        cmap_3d = [task_cmap[t] for t in aux_3d.index.get_level_values('Window Name').unique()]
    if not (aux_2d is None):
        aux_2d = aux_2d.reset_index()
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d_task_cmap = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
        
        si_Md_report = si_LE.loc['Procrustes',input_data,scenario,dist,knn,m].round(2)
        si_Md_report.loc['Subject','SI [3D]'] = load_LE_SI(['Procrustes'],input_datas=[input_data],dist_metrics=[dist],ms=[m],knns=[knn],norm_methods=[scenario],no_tqdm=True).set_index('Target').loc['Subject'].SI
        si_Md_report.loc['Task','SI [3D]'] = load_LE_SI(['Procrustes'],input_datas=[input_data],dist_metrics=[dist],ms=[m],knns=[knn],norm_methods=[scenario],no_tqdm=True).set_index('Target').loc['Window Name'].SI

    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_LE.loc['Procrustes',input_data,scenario,dist,knn,2].round(2),width=150),
                             pn.pane.DataFrame(si_LE.loc['Procrustes',input_data,scenario,dist,knn,3].round(2),width=150),
                             pn.pane.DataFrame(si_Md_report.round(2),width=150)],ncols=3)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_LE.loc['Procrustes',input_data,scenario,dist,knn,2].round(2),width=150),
                             pn.pane.DataFrame(si_LE.loc['Procrustes',input_data,scenario,dist,knn,3].round(3),width=150)],ncols=2)
    return plots

@pn.depends(sbj_select)
def le_scan_knn_plots(sbj):
    path_2d_plot = osp.join(PRJ_DIR,'Dashboard','Figures','LE','SIvsKNN_ScanLevel_{sbj}_m2_Task.png'.format(sbj=sbj))
    path_3d_plot = osp.join(PRJ_DIR,'Dashboard','Figures','LE','SIvsKNN_ScanLevel_{sbj}_m3_Task.png'.format(sbj=sbj))
    return pn.Row(pn.pane.PNG(path_2d_plot,height=200),pn.pane.PNG(path_3d_plot,height=200))


# +
le_figs_folder                = osp.join(PRJ_DIR,'Dashboard','Figures','LE')
le_config_card                = pn.Column(le_conf_box)
le_embs_scan_card             = pn.layout.Card(plot_LE_Scan_scats,title='Scatter Plots - One Scan', width=825)
le_embs_group_concat_card     = pn.layout.Card(plot_LE_Group_Concat_scats,title='Scatter Plots - Group Concatenation', width=825)
le_embs_group_procrustes_card = pn.layout.Card(plot_LE_Group_Procrustes_scats,title='Scatter Plots - Procrustes', width=825)
le_embs_row = pn.Row(le_embs_scan_card ,le_embs_group_concat_card,le_embs_group_procrustes_card)

le_knn_scan_avg_card   = pn.layout.Card(pn.Row(pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_ScanLevel_AVG_m2_Task.png'), height=200),
                                              pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_ScanLevel_AVG_m3_Task.png'), height=200)),title='Scan-Level Average | Summary View')
le_knn_scan_one_card   = pn.layout.Card(le_scan_knn_plots, title='This Scan | Summary View')
le_knn_group_conc_2d_card =  pn.layout.Card(pn.Row(pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_ALL_m2_Subject.png'), height=200),
                                                pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_ALL_m3_Subject.png')   , height=200)),title='Group-Level | Concatenation - SI[Subject]') 
le_knn_group_conc_3d_card =  pn.layout.Card(pn.Row(pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_ALL_m3_Task.png'), height=200),
                                                pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_ALL_m2_Task.png')   , height=200)),title='Group-Level | Concatenation - SI[Task]') 
le_knn_group_proc_2d_card =  pn.layout.Card(pn.Row(pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m2_Subject.png'), height=200),
                                                pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m3_Subject.png')   , height=200)),title='Group-Level | Procrustes - SI[Subject]') 
le_knn_group_proc_3d_card =  pn.layout.Card(pn.Row(pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m2_Task.png'), height=200),
                                                pn.pane.PNG(osp.join(le_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m3_Task.png')   , height=200)),title='Group-Level | Procrustes - SI[Task]') 

le_knn_plot_grid = pn.GridBox(*[le_knn_scan_one_card,le_knn_scan_avg_card,le_knn_group_conc_2d_card,le_knn_group_conc_3d_card,le_knn_group_proc_2d_card,le_knn_group_proc_3d_card],nrows=3)
# -

le_tab=pn.Column(pn.Row(le_config_card,le_knn_plot_grid),le_embs_row)

# + [markdown] tags=[]
# ***
# # UMAP
# #### 1. Load Silhouette Index for UMAP
#
# Hyper-parameter space: 3 Inputs * 2 Norm Approach * 8 m * 3 dist * x knns * 3 alphas = 
# * "Concat + UMAP": 17280 entries
# * "UMAP + Procrustes": 17280 entries
# * Single-Scan Level: 345600 entries
# -

si_UMAP = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_UMAP.pkl'))

REGENERATE_UMAP_KNN_PLOTS = False
if REGENERATE_UMAP_KNN_PLOTS:
    generate_Avg_UMAP_SIvsKnn_ScanLevel(si_UMAP,PNAS2015_subject_list)
    generate_UMAP_SIvsKNN_GroupLevel(si_UMAP)
    for sbj in PNAS2015_subject_list:
        generate_UMAP_SIvsKnn_ScanLevel(si_UMAP, sbj)

# #### 3. UMAP Tab Elements

# +
umap_figs_folder  = osp.join(PRJ_DIR,'Dashboard','Figures','UMAP')
umap_knn_select   = pn.widgets.Select(name='Knn',             options=umap_knns,         value=umap_knns[0], width=150)
umap_dist_select  = pn.widgets.Select(name='Distance Metric', options=umap_dist_metrics, value=umap_dist_metrics[0], width=150)
umap_m_select     = pn.widgets.Select(name='M',   options=umap_ms,           value=umap_ms[0], width=150)
umap_alpha_select = pn.widgets.Select(name='Learning Rate',   options=umap_alphas,       value=umap_alphas[0], width=150)
umap_init_select  = pn.widgets.Select(name='Init Method',     options=['spectral'],        value='spectral', width=150)
umap_mdist_select = pn.widgets.Select(name='Minimum Distance', options=[0.8],            value=0.8, width=150)
umap_grcc_col_sel = pn.widgets.Select(name='[G-CC] Color By:', options=['Window Name','Subject','Alertness','Focus','Consistency','Dificulty'], value='Window Name', width=150)
umap_grpt_col_sel = pn.widgets.Select(name='[G-PT] Color By:', options=['Window Name','Subject','Alertness','Focus','Consistency','Dificulty'], value='Window Name', width=150)
umap_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows', width=150)

umap_conf_box     = pn.WidgetBox(umap_dist_select,umap_knn_select,umap_init_select,umap_m_select,umap_alpha_select,umap_mdist_select,umap_grcc_col_sel,umap_grpt_col_sel,umap_drop_xxxx)

umap_LEFT_col     = pn.Column(umap_conf_box)
umap_LEFT_col


# -

@pn.depends(sbj_select,input_select,scenario_select,umap_dist_select,umap_knn_select,umap_m_select,umap_alpha_select,umap_init_select,umap_mdist_select,plot2d_toolbar_select,umap_drop_xxxx)
def plot_UMAP_Scan_scats(sbj,input_data,scenario,dist,knn,m,alpha,init_method,min_dist,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    if m == 2:
        aux_2d = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,m,drop_xxxx=drop_xxxx)
        
    # Preprare Embeddings and scales for plotting
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
        aux_3d_task_cmap = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
        
    
    # Plotting
    if (not (aux_2d is None)) & (aux_3d is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             pn.pane.DataFrame(si_UMAP.loc[sbj,input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150)],ncols=1)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_UMAP.loc[sbj,input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc[sbj,input_data,scenario,init_method,min_dist,dist,knn,alpha,3].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='UMAP001',y='UMAP002',z='UMAP003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_UMAP.loc[sbj,input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc[sbj,input_data,scenario,init_method,min_dist,dist,knn,alpha,3].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc[sbj,input_data,scenario,init_method,min_dist,dist,knn,alpha,m].round(2),width=150)
                            ],ncols=3)
    return plots            


@pn.depends(input_select,scenario_select,umap_dist_select,umap_knn_select,umap_m_select,umap_alpha_select,umap_init_select,umap_mdist_select,umap_grcc_col_sel,plot2d_toolbar_select,umap_drop_xxxx)
def plot_UMAP_Group_Concat_scats(input_data,scenario,dist,knn,m,alpha,init_method,min_dist,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    if m == 2:
        aux_2d = load_single_umap('ALL',input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_umap('ALL',input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap('ALL',input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_umap('ALL',input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap('ALL',input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_umap('ALL',input_data,scenario,dist,knn,alpha,init_method,min_dist,m,drop_xxxx=drop_xxxx)
    # Preprare Embeddings
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    # Prepare Color-scales
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    #Plotting
    if (not (aux_2d is None)) & (aux_3d is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             pn.pane.DataFrame(si_UMAP.loc['ALL',input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150)],ncols=1)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_UMAP.loc['ALL',input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc['ALL',input_data,scenario,init_method,min_dist,dist,knn,alpha,3].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_UMAP.loc['ALL',input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc['ALL',input_data,scenario,init_method,min_dist,dist,knn,alpha,3].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc['ALL',input_data,scenario,init_method,min_dist,dist,knn,alpha,m].round(2),width=150)
                            ],ncols=3) 
    return plots


@pn.depends(input_select,scenario_select,umap_dist_select,umap_knn_select,umap_m_select,umap_alpha_select,umap_init_select,umap_mdist_select,umap_grpt_col_sel,plot2d_toolbar_select,umap_drop_xxxx)
def plot_UMAP_Group_Procustes_scats(input_data,scenario,dist,knn,m,alpha,init_method,min_dist,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    if m == 2:
        aux_2d = load_single_umap('Procrustes',input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_umap('Procrustes',input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap('Procrustes',input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_umap('Procrustes',input_data,scenario,dist,knn,alpha,init_method,min_dist,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_umap('Procrustes',input_data,scenario,dist,knn,alpha,init_method,min_dist,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_umap('Procrustes',input_data,scenario,dist,knn,alpha,init_method,min_dist,m,drop_xxxx=drop_xxxx)
    # Preprare Embeddings
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    # Prepare Color-scales
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    #Plotting
    if (not (aux_2d is None)) & (aux_3d is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             pn.pane.DataFrame(si_UMAP.loc['Procrustes',input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150)],ncols=1)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_UMAP.loc['Procrustes',input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc['Procrustes',input_data,scenario,init_method,min_dist,dist,knn,alpha,3].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='UMAP001',y='UMAP002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='UMAP001',y='UMAP002',z='UMAP003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_UMAP.loc['Procrustes',input_data,scenario,init_method,min_dist,dist,knn,alpha,2].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc['Procrustes',input_data,scenario,init_method,min_dist,dist,knn,alpha,3].round(2),width=150),
                             pn.pane.DataFrame(si_UMAP.loc['Procrustes',input_data,scenario,init_method,min_dist,dist,knn,alpha,m].round(2),width=150)
                            ],ncols=3) 
    return plots


@pn.depends(scenario_select)
def umap_scan_avg_nm_show(nm):
    plot1 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_ScanLevel_AVG_m2_{nm}_Task.png'.format(nm=nm)), height=200)
    plot2 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_ScanLevel_AVG_m3_{nm}_Task.png'.format(nm=nm)), height=200)
    return pn.Row(plot1,plot2)
umap_knn_scan_avg_card = pn.layout.Card(umap_scan_avg_nm_show,title='Scan-Level Average | Summary View')
#umap_knn_scan_avg_card   = pn.layout.Card(pn.Row(umap_scan_avg_nm_select,umap_scan_avg_nm_show),title='Scan-Level Average | Summary View')

@pn.depends(sbj_select,scenario_select)
def umap_scan_sbj_nm_show(sbj,nm):
    plot1 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_ScanLevel_{sbj}_m2_{nm}_Task.png'.format(nm=nm,sbj=sbj)), height=200)
    plot2 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_ScanLevel_{sbj}_m3_{nm}_Task.png'.format(nm=nm,sbj=sbj)), height=200)
    return pn.Row(plot1,plot2)
#umap_knn_scan_sbj_card   = pn.layout.Card(pn.Row(umap_scan_sbj_nm_select,umap_scan_sbj_nm_show),title='This Scan | Summary View')
umap_knn_scan_sbj_card   = pn.layout.Card(umap_scan_sbj_nm_show,title='This Scan | Summary View')


@pn.depends(scenario_select)
def umap_knn_group_conc_sbj_plots(nm):
    plot1 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_ALL_m2_{nm}_Subject.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_ALL_m3_{nm}_Subject.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
umap_knn_group_conc_sbj_card = pn.layout.Card(umap_knn_group_conc_sbj_plots,title='Group-Level | Concatenation - SI[Subject]')
@pn.depends(scenario_select)
def umap_knn_group_conc_task_plots(nm):
    plot1 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_ALL_m2_{nm}_Task.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_ALL_m3_{nm}_Task.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
umap_knn_group_conc_task_card = pn.layout.Card(umap_knn_group_conc_task_plots,title='Group-Level | Concatenation - SI[Task]')


@pn.depends(scenario_select)
def umap_knn_group_procrustes_sbj_plots(nm):
    plot1 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m2_{nm}_Subject.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m3_{nm}_Subject.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
umap_knn_group_procrustes_sbj_card = pn.layout.Card(umap_knn_group_procrustes_sbj_plots,title='Group-Level | Procrustes - SI[Subject]')
@pn.depends(scenario_select)
def umap_knn_group_procrustes_task_plots(nm):
    plot1 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m2_{nm}_Task.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(umap_figs_folder,'SIvsKNN_GroupLevel_Procrustes_m3_{nm}_Task.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
umap_knn_group_procrustes_task_card = pn.layout.Card(umap_knn_group_procrustes_task_plots,title='Group-Level | Procrustes - SI[Task]')

umap_knn_plot_grid = pn.GridBox(*[umap_knn_scan_sbj_card,umap_knn_scan_avg_card,
                                 umap_knn_group_conc_sbj_card,umap_knn_group_conc_task_card,
                                 umap_knn_group_procrustes_sbj_card,umap_knn_group_procrustes_task_card],ncols=2)

umap_knn_plot_grid = pn.GridBox(*[umap_knn_scan_sbj_card,umap_knn_scan_avg_card,
                                 umap_knn_group_conc_sbj_card,umap_knn_group_conc_task_card,
                                 umap_knn_group_procrustes_sbj_card,umap_knn_group_procrustes_task_card],ncols=2)

umap_embs_scan_card             = pn.layout.Card(plot_UMAP_Scan_scats,title='Scatter Plots - One Scan', width=825)
umap_embs_group_concat_card     = pn.layout.Card(plot_UMAP_Group_Concat_scats,title='Scatter Plots - Group Concatenation', width=825)
umap_embs_group_procrustes_card = pn.layout.Card(plot_UMAP_Group_Procustes_scats,title='Scatter Plots - Procrustes', width=825)
umap_embs_row = pn.Row(umap_embs_scan_card ,umap_embs_group_concat_card,umap_embs_group_procrustes_card)

umap_tab = pn.Column(pn.Row(umap_LEFT_col,umap_knn_plot_grid),umap_embs_row)

# + [markdown] tags=[]
# ***
# # TSNE
# #### 1. Load Silhouette Index for TSNE

# + tags=[]
# %%time
RELOAD_SI_TSNE = False
if RELOAD_SI_TSNE:
    print('++ INFO: Loading SI for Concat + TSNE....')
    si_TSNE_all              = load_TSNE_SI(sbj_list=['ALL'],               check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    print('++ INFO: Loading SI for TSNE + Procrustes....')
    si_TSNE_procrustes       = load_TSNE_SI(sbj_list=['Procrustes'],        check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_TSNE_procrustes_extra = load_TSNE_SI(sbj_list=['Procrustes'],        check_availability=False, verbose=False, wls=wls, wss=wss, ms=[5,10], dist_metrics=['correlation'], input_datas=['Original'], alphas=[10,1000])
    print('++ INFO: Loading SI for scan level TSNE...')
    si_TSNE_scans            = load_TSNE_SI(sbj_list=PNAS2015_subject_list, check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_TSNE_scans_extra      = load_TSNE_SI(sbj_list=PNAS2015_subject_list, check_availability=False, verbose=False, wls=wls, wss=wss, ms=[5,10], dist_metrics=['correlation'], input_datas=['Original'], alphas=[10,1000])

    si_TSNE = pd.concat([si_TSNE_scans, si_TSNE_all, si_TSNE_procrustes, si_TSNE_procrustes_extra])
    si_TSNE.replace('Window Name','Task', inplace=True)
    si_TSNE = si_TSNE.set_index(['Subject','Input Data','Norm','Metric','PP','m','Alpha','Init','Target']).sort_index()
    del si_TSNE_scans, si_TSNE_all, si_TSNE_procrustes, si_TSNE_procrustes_extra, si_TSNE_scans_extra
    
    si_TSNE.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))
else:
    si_TSNE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))

# +
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
HUGE_SIZE   = 22

NULL_CONNRAND_PALETTE  = sns.color_palette('Wistia',n_colors=3)
NULL_PHASERAND_PALETTE = sns.color_palette('gist_gray',n_colors=3)
ORIGINAL_PALETTE       = sns.color_palette(palette='bright',n_colors=3)
from utils.basics import PRJ_DIR, group_method_2_label

def generate_TSNE_SIvsKNN_GroupLevel(si,comb_methods=['ALL','Procrustes'],figsize=(10,5),target_m_tuples=[('Task',2),('Task',3),('Subject',2),('Subject',3)], 
                                     verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200,init_method='pca', norm_methods=norm_methods):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for norm_method in norm_methods:
        for group_method in comb_methods:
            for (target,m) in target_m_tuples:
                fig, ax = plt.subplots(1,1,figsize=figsize)
                data_orig = si.loc[group_method].loc['Original',norm_method,:,:,m,:,init_method,target]
                data_nc = si.loc[group_method].loc['Null_ConnRand',norm_method,:,:,m,:,init_method,target]
                data_np = si.loc[group_method].loc['Null_PhaseRand',norm_method,:,:,m,:,init_method,target]

                sns.lineplot(data=data_orig,y='SI',x='PP', hue='Metric', hue_order=['correlation','cosine','euclidean'],ax=ax, palette=ORIGINAL_PALETTE)
                sns.lineplot(data=data_nc,y='SI',x='PP',hue='Metric', hue_order=['correlation','cosine','euclidean'], legend=False,ax=ax, palette=NULL_CONNRAND_PALETTE)
                sns.lineplot(data=data_np,y='SI',x='PP',hue='Metric', hue_order=['correlation','cosine','euclidean'], legend=False,ax=ax, palette=NULL_PHASERAND_PALETTE)
                ax.set_ylim(y_min,y_max)
                ax.set_xlim(x_min,x_max)
                ax.grid()
                ax.set_title('Group-Level [{gm}] | {m}D'.format(m=str(m),gm=group_method_2_label[group_method]), fontsize=HUGE_SIZE)
                ax.legend(loc='upper right', ncol=2)
                ax.set_ylabel('$SI_{%s}$' % target)
                #ax.set_ylabel('SI [{target}]'.format(target=target))
                out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','TSNE')
                if not osp.exists(out_dir):
                    os.makedirs(out_dir)
                    if verbose:
                        print('++ INFO: Folder created [%s]' % out_dir)
                out_path = osp.join(out_dir,'SIvsPP_GroupLevel_{gm}_m{m}_{nm}_{target}.png'.format(m=str(m), target=target, gm=group_method, nm=norm_method))
                plt.savefig(out_path,bbox_inches='tight')
                if verbose:
                    print('++ INFO: Figure saved to disk [%s]' % out_path)
                print(group_method,norm_method,m,init_method,target)
                asdad
                plt.close()


# -

# #### 2. Generate Summary View Figures

# %%time
REGENERATE_TSNE_KNN_PLOTS = False
if REGENERATE_TSNE_KNN_PLOTS:
    generate_Avg_TSNE_SIvsKnn_ScanLevel(si_TSNE,PNAS2015_subject_list)
    generate_TSNE_SIvsKNN_GroupLevel(si_TSNE)
    for sbj in PNAS2015_subject_list:
        generate_TSNE_SIvsKnn_ScanLevel(si_TSNE, sbj)

# #### 3.TSNE Tab Elements

# +
tsne_figs_folder  = osp.join(PRJ_DIR,'Dashboard','Figures','TSNE')
tsne_pp_select   = pn.widgets.Select(name='Perplexity',        options=tsne_pps,          value=tsne_pps[0], width=150)
tsne_dist_select  = pn.widgets.Select(name='Distance Metric',  options=tsne_dist_metrics, value=tsne_dist_metrics[0], width=150)
tsne_m_select     = pn.widgets.Select(name='M',                options=[2,3,5,10],       value=2, width=150)
tsne_alpha_select = pn.widgets.Select(name='Learning Rate',    options=tsne_alphas,       value=tsne_alphas[0], width=150)
tsne_init_select  = pn.widgets.Select(name='Init Method',      options=tsne_inits,       value=tsne_inits[0], width=150)
tsne_grcc_col_sel = pn.widgets.Select(name='[G-CC] Color By:', options=['Window Name','Subject'], value='Window Name', width=150)
tsne_grpt_col_sel = pn.widgets.Select(name='[G-PT] Color By:', options=['Window Name','Subject'], value='Window Name', width=150)
tsne_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows', width=150)

tsne_conf_box     = pn.WidgetBox(tsne_dist_select,tsne_pp_select,tsne_init_select,tsne_m_select,tsne_alpha_select,tsne_grcc_col_sel,tsne_grpt_col_sel,tsne_drop_xxxx)

tsne_LEFT_col     = pn.Column(tsne_conf_box)


# -

@pn.depends(sbj_select,input_select,scenario_select,tsne_dist_select,tsne_pp_select,tsne_m_select,tsne_alpha_select,tsne_init_select,plot2d_toolbar_select,tsne_drop_xxxx)
def plot_TSNE_Scan_scats(sbj,input_data,scenario,dist,pp,m,alpha,init_method,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    if m == 2:
        aux_2d = load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
    elif m == 3:
        aux_2d = load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
    else:
        aux_2d = load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        aux_3d = load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
        aux_Md = load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,m,drop_xxxx=drop_xxxx)
    # Preprare Embeddings and scales for plotting
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)

    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
        aux_3d_task_cmap = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)

   
    # Plotting
    if (not (aux_2d is None)) & (aux_3d is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             pn.pane.DataFrame(si_TSNE.loc[sbj,input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150)],ncols=1)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_TSNE.loc[sbj,input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc[sbj,input_data,scenario,dist,pp,3,alpha,init_method].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c='Window Name', cmap=task_cmap, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='TSNE001',y='TSNE002',z='TSNE003',c='Window Name', cmap=aux_3d_task_cmap,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_TSNE.loc[sbj,input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc[sbj,input_data,scenario,dist,pp,3,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc[sbj,input_data,scenario,dist,pp,m,alpha,init_method].round(2),width=150)
                            ],ncols=3)
    return plots


@pn.depends(input_select,scenario_select,tsne_dist_select,tsne_pp_select,tsne_m_select,tsne_alpha_select,tsne_init_select,tsne_grpt_col_sel,plot2d_toolbar_select,tsne_drop_xxxx)
def plot_TSNE_Group_Procustes_scats(input_data,scenario,dist,pp,m,alpha,init_method,color_col,plot_2d_toolbar, drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    if m == 2:
        aux_2d = load_single_tsne('Procrustes',input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_2d.columns)) & ('Subject' in list(aux_2d.columns)):
            aux_2d = aux_2d.set_index(['Window Name','Subject'])
    elif m == 3:
        aux_2d = load_single_tsne('Procrustes',input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_2d.columns)) & ('Subject' in list(aux_2d.columns)):
            aux_2d = aux_2d.set_index(['Window Name','Subject'])
        aux_3d = load_single_tsne('Procrustes',input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_3d.columns)) & ('Subject' in list(aux_3d.columns)):
            aux_3d = aux_3d.set_index(['Window Name','Subject'])
    else:
        aux_2d = load_single_tsne('Procrustes',input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_2d.columns)) & ('Subject' in list(aux_2d.columns)):
            aux_2d = aux_2d.set_index(['Window Name','Subject'])
        aux_3d = load_single_tsne('Procrustes',input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_3d.columns)) & ('Subject' in list(aux_3d.columns)):
            aux_3d = aux_3d.set_index(['Window Name','Subject'])
        aux_Md = load_single_tsne('Procrustes',input_data,scenario,dist,pp,alpha,init_method,m,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_Md.columns)) & ('Subject' in list(aux_Md.columns)):
            aux_Md = aux_Md.set_index(['Window Name','Subject'])
    # Preprare Embeddings
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    
    # Prepare Color-scales
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    
    #PLOT - CONTINUE HERE
    # Plotting
    if (not (aux_2d is None)) & (aux_3d is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             pn.pane.DataFrame(si_TSNE.loc['Procrustes',input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150)],ncols=1)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_TSNE.loc['Procrustes',input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc['Procrustes',input_data,scenario,dist,pp,3,alpha,init_method].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_TSNE.loc['Procrustes',input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc['Procrustes',input_data,scenario,dist,pp,3,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc['Procrustes',input_data,scenario,dist,pp,m,alpha,init_method].round(2),width=150)
                            ],ncols=3)
    return plots


@pn.depends(input_select,scenario_select,tsne_dist_select,tsne_pp_select,tsne_m_select,tsne_alpha_select,tsne_init_select,tsne_grcc_col_sel,plot2d_toolbar_select,tsne_drop_xxxx)
def plot_TSNE_Group_Concat_scats(input_data,scenario,dist,pp,m,alpha,init_method,color_col,plot_2d_toolbar,drop_xxxx):
    plots = None
    aux_2d, aux_3d, aux_Md = None, None, None
    # Load all necessary embeddings
    if m == 2:
        aux_2d = load_single_tsne('ALL',input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_2d.columns)) & ('Subject' in list(aux_2d.columns)):
            aux_2d = aux_2d.set_index(['Window Name','Subject'])
    elif m == 3:
        aux_2d = load_single_tsne('ALL',input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_2d.columns)) & ('Subject' in list(aux_2d.columns)):
            aux_2d = aux_2d.set_index(['Window Name','Subject'])
        aux_3d = load_single_tsne('ALL',input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_3d.columns)) & ('Subject' in list(aux_3d.columns)):
            aux_3d = aux_3d.set_index(['Window Name','Subject'])
    else:
        aux_2d = load_single_tsne('ALL',input_data,scenario,dist,pp,alpha,init_method,2,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_2d.columns)) & ('Subject' in list(aux_2d.columns)):
            aux_2d = aux_2d.set_index(['Window Name','Subject'])
        aux_3d = load_single_tsne('ALL',input_data,scenario,dist,pp,alpha,init_method,3,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_3d.columns)) & ('Subject' in list(aux_3d.columns)):
            aux_3d = aux_3d.set_index(['Window Name','Subject'])
        aux_Md = load_single_tsne('ALL',input_data,scenario,dist,pp,alpha,init_method,m,drop_xxxx=drop_xxxx)
        if ('Window Name' in list(aux_Md.columns)) & ('Subject' in list(aux_Md.columns)):
            aux_Md = aux_Md.set_index(['Window Name','Subject'])
    # Preprare Embeddings
    if not (aux_2d is None):
        aux_2d = aux_2d.apply(zscore)
        aux_2d = aux_2d.reset_index()
        aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_3d is None):
        aux_3d = aux_3d.apply(zscore)
        aux_3d = aux_3d.reset_index()
        aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    if not (aux_Md is None):
        aux_Md = aux_Md.apply(zscore)
        aux_Md = aux_Md.reset_index()
        aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)
    
    # Prepare Color-scales
    if color_col == 'Subject':
        cmap_2d = sbj_cmap_dict
        cmap_3d = sbj_cmap_list
    else:
        cmap_2d = task_cmap
        if not(aux_3d is None):
            cmap_3d = [task_cmap[t] for t in aux_3d['Window Name'].unique()]
    
    #PLOT - CONTINUE HERE
    # Plotting
    if (not (aux_2d is None)) & (aux_3d is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             pn.pane.DataFrame(si_TSNE.loc['ALL',input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150)],ncols=1)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_TSNE.loc['ALL',input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc['ALL',input_data,scenario,dist,pp,3,alpha,init_method].round(2),width=150)
                            ],ncols=2)
    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):
        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='TSNE001',y='TSNE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),
                             plot_3d_scatter(aux_3d,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             plot_3d_scatter(aux_Md,x='TSNE001',y='TSNE002',z='TSNE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),
                             pn.pane.DataFrame(si_TSNE.loc['ALL',input_data,scenario,dist,pp,2,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc['ALL',input_data,scenario,dist,pp,3,alpha,init_method].round(2),width=150),
                             pn.pane.DataFrame(si_TSNE.loc['ALL',input_data,scenario,dist,pp,m,alpha,init_method].round(2),width=150)
                            ],ncols=3)
    return plots


# +
@pn.depends(scenario_select)
def tsne_scan_avg_nm_show(nm):
    plot1 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_ScanLevel_AVG_m2_{nm}_Task.png'.format(nm=nm)), height=200)
    plot2 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_ScanLevel_AVG_m3_{nm}_Task.png'.format(nm=nm)), height=200)
    return pn.Row(plot1,plot2)
tsne_pp_scan_avg_card = pn.layout.Card(tsne_scan_avg_nm_show,title='Scan-Level Average | Summary View')

@pn.depends(sbj_select,scenario_select)
def tsne_scan_sbj_nm_show(sbj,nm):
    plot1 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_ScanLevel_{sbj}_m2_{nm}_Task.png'.format(nm=nm,sbj=sbj)), height=200)
    plot2 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_ScanLevel_{sbj}_m3_{nm}_Task.png'.format(nm=nm,sbj=sbj)), height=200)
    return pn.Row(plot1,plot2)
tsne_pp_scan_sbj_card   = pn.layout.Card(tsne_scan_sbj_nm_show,title='This Scan | Summary View')


# -

@pn.depends(scenario_select)
def tsne_pp_group_conc_sbj_plots(nm):
    plot1 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_ALL_m2_{nm}_Subject.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_ALL_m3_{nm}_Subject.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
tsne_pp_group_conc_sbj_card = pn.layout.Card(tsne_pp_group_conc_sbj_plots,title='Group-Level | Concatenation - SI[Subject]')
@pn.depends(scenario_select)
def tsne_pp_group_conc_task_plots(nm):
    plot1 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_ALL_m2_{nm}_Task.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_ALL_m3_{nm}_Task.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
tsne_pp_group_conc_task_card = pn.layout.Card(tsne_pp_group_conc_task_plots,title='Group-Level | Concatenation - SI[Task]')


@pn.depends(scenario_select)
def tsne_pp_group_procrustes_sbj_plots(nm):
    plot1 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_Procrustes_m2_{nm}_Subject.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_Procrustes_m3_{nm}_Subject.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
tsne_pp_group_procrustes_sbj_card = pn.layout.Card(tsne_pp_group_procrustes_sbj_plots,title='Group-Level | Procrustes - SI[Subject]')
@pn.depends(scenario_select)
def tsne_pp_group_procrustes_task_plots(nm):
    plot1 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_Procrustes_m2_{nm}_Task.png').format(nm=nm), height=200)
    plot2 = pn.pane.PNG(osp.join(tsne_figs_folder,'SIvsPP_GroupLevel_Procrustes_m3_{nm}_Task.png').format(nm=nm), height=200)
    return pn.Row(plot1,plot2)
tsne_pp_group_procrustes_task_card = pn.layout.Card(tsne_pp_group_procrustes_task_plots,title='Group-Level | Procrustes - SI[Task]')

tsne_pp_plot_grid = pn.GridBox(*[tsne_pp_scan_sbj_card,tsne_pp_scan_avg_card,
                                 tsne_pp_group_conc_sbj_card,tsne_pp_group_conc_task_card,
                                 tsne_pp_group_procrustes_sbj_card,tsne_pp_group_procrustes_task_card],ncols=2)

tsne_embs_scan_card             = pn.layout.Card(plot_TSNE_Scan_scats,title='Scatter Plots - One Scan', width=825)
tsne_embs_group_concat_card     = pn.layout.Card(plot_TSNE_Group_Concat_scats,title='Scatter Plots - Group Concatenation', width=825)
tsne_embs_group_procrustes_card = pn.layout.Card(plot_TSNE_Group_Procustes_scats,title='Scatter Plots - Procrustes', width=825)
tsne_embs_row = pn.Row(tsne_embs_scan_card,tsne_embs_group_concat_card,tsne_embs_group_procrustes_card)

tsne_tab = pn.Column(pn.Row(tsne_LEFT_col,tsne_pp_plot_grid),tsne_embs_row)

# + tags=[]
dashboard = pn.Column(data_select_box,pn.Tabs(('Laplacian Eigenmaps',le_tab),('TSNE',tsne_tab),('UMAP',umap_tab)))

# +
#dashboard = pn.Column(data_select_box,pn.Tabs(('Laplacian Eigenmaps',le_tab)))
# -

dashboard_server = dashboard.show(port=port_tunnel,open=False)

dashboard_server.stop()
