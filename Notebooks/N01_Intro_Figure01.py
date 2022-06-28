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

# # Figure 1. Introduction to low dimensional representations of tvFC data
#
# This notebook will generate all the elements in Figure 1. To do that it requires three files made available in ```Resources/Figure01```:
#
# * ```winlabels_wl030_ws001.csv```: This file contains the information regarding what task was performed during each window. It contains one string (e.g., REST, VIDEO, etc) per line and has as many lines as windows.
# * ```ROI_Coordinates.txt```: This file contains the coordinates for the centroid of each ROI. We need this information to sort connections on the basis of hemispheric membership.
# * ```sbj06_ctask001_nroi0200_wl030_ws001.csv```: This file contains the representative timeseries for one multi-task scan.

from scipy.io import loadmat
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import seaborn as sns
import plotly.express as px
from matplotlib.patches import Rectangle
pn.extension('plotly')

from utils.data_functions import compute_SWC
from utils.basics import PRJ_DIR, task_cmap

# In order to compute the Sliding Window Connectivity matrix, we need to know the repetition time (TR), the window duration and the window step. Those are defined below

TR      = 1.5     # TR = 1.5 sec
WL_secs = 45      # Window length in seconds
WL_trs  = int(WL_secs/TR)
WS_trs  = 1.5

# Creates variable with the path where the necessary files reside

fig01_resource_folder = osp.join(PRJ_DIR,'Resources','Figure01')

# These dictionaries will be used to atomatically set some plotting options for Plotly 3D scatter plots

camera = dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=1))
scene_correct_le = dict(
        xaxis = dict(nticks=4, range=[-.005,.005], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=4, range=[-.005,.005], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=4, range=[-.005,.005], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))
scene_incorrect_le = dict(
        xaxis = dict(nticks=4, range=[-.02,.02], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=4, range=[-.02,.02], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=4, range=[-.02,.02], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))

# *** 
# ## 1. Load ROI timeseries for one scan

roi_ts = pd.read_csv(osp.join(fig01_resource_folder,'sbj06_ctask001_nroi0200_wl030_ws001.csv'), index_col=[0])

print('++ INFO: Time series dataframe shape: % s' % str(roi_ts.shape))
roi_ts.head(5)

# ***
# ## 2. Compute Sliding Window Correlation

win_labels = np.loadtxt('/data/SFIMJGC_HCP7T/manifold_learning_fmri/Resources/Figure01/winlabels_wl030_ws001.csv', dtype='str')

# %%time
swc_r,swc_Z, winInfo = compute_SWC(roi_ts,WL_trs,WS_trs,win_names=win_labels,window=None)
swc_r.index.name = 'Connections'
swc_Z.index.name = 'Connections'
print("++ INFO: Size of sliding window correlation: %s" % str(swc_r.shape))

swc_r.to_csv(osp.join(PRJ_DIR,'Resources','Figure03','swcR_sbj06_ctask001_nroi0200_wl030_ws001.csv'), float_format='%.2f')
swc_Z.to_csv(osp.join(PRJ_DIR,'Resources','Figure03','swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv'), float_format='%.2f')

swc_r

# For plotting purposes, we need to know what task was being performed during each window. This information is loaded now into the win_labels variable.

# ## 3. Compute Different Matrix Sortings
#
# In Figure 1, we will plot the same tvFC matrix sorting connections in three different ways: 1) in descending order of mean FC, 2) in descencing order of FC volatility (as indexed by the coefficient of variance), and 3) based on hemispheric membership. The next few cells in this section will generate the appropriate indexes with each of these sorting schemes.
#
# First, we do 1) mean and 2) volatility

volatility_sorting_idx = (swc_r.std(axis=1)/swc_r.mean(axis=1).abs()).sort_values(ascending=False).index
mean_sorting_idx       = swc_r.mean(axis=1).sort_values(ascending=False).index

# Next, we also compute sorting of connections based on hemispheric membership

ROI_CMs = pd.read_csv(osp.join(fig01_resource_folder,'ROI_Coordinates.txt'), index_col='ROI_ID')

con_info = pd.DataFrame(index=swc_r.index,columns=['Hemi_A','Hemi_B'])
for i,j in swc_r.index:
    if ROI_CMs.loc[i,'x']<0:
        con_info.loc[(i,j),'Hemi_A'] = 'R'
    else:
        con_info.loc[(i,j),'Hemi_A'] = 'L'
    if ROI_CMs.loc[j,'x']<0:
        con_info.loc[(i,j),'Hemi_B'] = 'R'
    else:
        con_info.loc[(i,j),'Hemi_B'] = 'L'

LL_Cons = con_info[(con_info['Hemi_A']=='L') & (con_info['Hemi_B']=='L')].index
RR_Cons = con_info[(con_info['Hemi_A']=='R') & (con_info['Hemi_B']=='R')].index
LR_Cons = con_info[(con_info['Hemi_A']=='L') & (con_info['Hemi_B']=='R')].index
RL_Cons = con_info[(con_info['Hemi_A']=='R') & (con_info['Hemi_B']=='L')].index

# ***
# ## 4. Show SWC Matrix at scale
#
# This is panel A in figure 1.

fig,ax = plt.subplots(1,1, figsize=(10,10))
sns.heatmap(swc_r.loc[mean_sorting_idx],cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False, square=True, cbar=False)
ax.set(ylabel=None);

# ***
# ## 5. Show SWC matrix with different sortings
#
# The following images correspond to panels (B), (C) and (D) in Figure 1.

plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(3,1,figsize=(30,30))
for i,(df,title) in enumerate(zip([swc_r.loc[mean_sorting_idx],swc_r.loc[volatility_sorting_idx],pd.concat([swc_r.loc[LL_Cons],swc_r.loc[RR_Cons],swc_r.loc[LR_Cons],swc_r.loc[RL_Cons]])],
                                ['Average Strength','Volatility','Hemisphere'])):
    plot1    = sns.heatmap(df, ax=axs[i],cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False)
    plot1.set_xlabel('Time [Windows]', fontsize=14)
    plot1.set_ylabel('Connections',    fontsize=14)
    #plot1.set_title(title, fontsize=14)
    plot1.add_patch(Rectangle((0, -301),          91, 300, fill=True, color=task_cmap['Rest'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+27, -301),     101, 300, fill=True, color=task_cmap['Memory'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+(2*27)+101, -301), 101, 300, fill=True, color=task_cmap['Video'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+(3*27)+(2*101), -301), 101, 300, fill=True, color=task_cmap['Math'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+(4*27)+(3*101), -301), 101, 300, fill=True, color=task_cmap['Memory'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+(5*27)+(4*101), -301), 101, 300, fill=True, color=task_cmap['Rest'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+(6*27)+(5*101), -301), 101, 300, fill=True, color=task_cmap['Math'], lw=0, clip_on=False))
    plot1.add_patch(Rectangle((91+(7*27)+(6*101), -301), 101, 300, fill=True, color=task_cmap['Video'], lw=0, clip_on=False))
    if i == 2:
        axs[i].hlines([LL_Cons.shape[0],LL_Cons.shape[0]+RR_Cons.shape[0]],*axs[i].get_xlim(), colors='w', linestyles='dashed')

# ***
# ## 6. Generate LE
#
# The last four panels of figure 1 depict a two low dimensional representations of this data generated with Laplacian Eigenmaps:
#
# * A behaviorally informative embedding generated using appropriate hyperparameters (panels E,F & G). The same low dimensional representation of the data is depicted with three different coloring schemes:
#     * one color
#     * colors according to task
#     * colors according to time
# * A non-informative embedding using an excessively low knn value.

# LE representations are generated using the scikit-learn library. The necessary functions are imported next

from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph

# We now generate an Spectral Embedding object that will generate a 3D represenation (n_components=3) and will expect a pre-computed affinity matrix. We set the random_seed to 43 for reproducibility of results

# Create Embedding Object
LE_obj     = SpectralEmbedding(n_components=3, affinity='precomputed', n_jobs=-1, random_state=43)

# First, we create a meaningful embedding using knn=90 duirng the computation of the affinity matrix. In addition to the 3D coordinates, we add two additional columns to the resulting panda object. One contains the task information and another one timing information. These two columns are only used for plotting purposes, but do not contribute in any way to the generation of the embeddings.

# Create Affinity Matrix with valid neighborhood size
X_affinity_correct = pd.DataFrame(kneighbors_graph(swc_Z.T, 90, include_self=False, n_jobs=-1, metric='correlation', mode='connectivity').toarray())
X_affinity_correct = ((0.5 * (X_affinity_correct + X_affinity_correct.T)) > 0).astype(int)
# Compute Embedding based on valid neiighborhood size
LE_correct         = pd.DataFrame(LE_obj.fit_transform(X_affinity_correct),columns=['Dim_'+str(i+1).zfill(2) for i in np.arange(3)])
LE_correct['Task'] = win_labels
LE_correct['Time'] = np.arange(LE_correct.shape[0])+1

# Finally, we also create a not so meaningful embedding using a excessively low knn value.

# Create Affinity Matrix with valid neighborhood size
X_affinity_incorrect = pd.DataFrame(kneighbors_graph(swc_Z.T, 5, include_self=False, n_jobs=-1, metric='correlation', mode='connectivity').toarray())
X_affinity_incorrect = ((0.5 * (X_affinity_incorrect + X_affinity_incorrect.T)) > 0).astype(int)
# Compute Embedding based on valid neiighborhood size
LE_incorrect         = pd.DataFrame(LE_obj.fit_transform(X_affinity_incorrect),columns=['Dim_'+str(i+1).zfill(2) for i in np.arange(3)])
LE_incorrect['Task'] = win_labels
LE_incorrect['Time'] = np.arange(LE_incorrect.shape[0])+1

# Here, we now plot the different versions of these two embeddings

fig_nocolor = px.scatter_3d(LE_correct,x='Dim_01',y='Dim_02',z='Dim_03', width=500, height=500, opacity=0.5)
fig_nocolor.update_layout(scene_camera=camera, scene=scene_correct_le,scene_aspectmode='cube',margin=dict(l=0, r=0, b=0, t=0));
fig_time = px.scatter_3d(LE_correct,x='Dim_01',y='Dim_02',z='Dim_03', width=500, height=500, opacity=0.5, color='Time',color_continuous_scale='twilight')
fig_time.update_layout(scene_camera=camera, scene=scene_correct_le,scene_aspectmode='cube',margin=dict(l=0, r=0, b=0, t=0));
fig_task = px.scatter_3d(LE_correct,x='Dim_01',y='Dim_02',z='Dim_03', width=500, height=500, opacity=0.5, color='Task',color_discrete_sequence=['gray','black','blue','green','yellow'])
fig_task.update_layout(scene_camera=camera, scene=scene_correct_le,scene_aspectmode='cube',margin=dict(l=0, r=0, b=0, t=0));
fig_task_bad = px.scatter_3d(LE_incorrect,x='Dim_01',y='Dim_02',z='Dim_03', width=500, height=500, opacity=0.5, color='Task',color_discrete_sequence=['gray','black','blue','green','yellow'])
fig_task_bad.update_layout(scene_camera=camera,scene=scene_incorrect_le,scene_aspectmode='cube',margin=dict(l=0, r=0, b=0, t=0));

pn.Column(pn.Row(pn.pane.Plotly(fig_nocolor),pn.pane.Plotly(fig_time)),
          pn.Row(pn.pane.Plotly(fig_task),pn.pane.Plotly(fig_task_bad)))

# ***
# # END OF NOTEBOOK
# ***
# The remaining of this notebook contains code that we used to generate the files now directly available in the Resources folder. 
#
# We keep this code here for internal purposes, but should not be needed to reproduce the results shown above this cell.

# ### Determine Hemispheric Location of each ROI
# To determine hemispheric location for each ROI, we need to run the following code
# ```bash
# # cd /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001
#
# # echo "ROI_ID,x,y,z" > ./ROI_Coordinates.txt
# for t in `seq 0 1 156`
# do
#     xyz=`3dCM SBJ06_CTask001.Craddock_T2Level_0200.lowSigma+orig[${t}]`
#     x=`echo ${xyz} | awk '{print$1}'`
#     y=`echo ${xyz} | awk '{print$2}'`
#     z=`echo ${xyz} | awk '{print$3}'`
#     roiid=`printf "%03d" ${t}`
#     echo "ROI${roiid},${x},${y},${z}" >> ./ROI_Coordinates.txt
# done
#
# ```

# ### Selection of Subject

SBJ     = 'SBJ06' # Subject number
TR      = 1.5     # TR = 1.5 sec
WL_secs = 45      # Window length in seconds
WL_trs  = int(WL_secs/TR)
WS_trs  = 10

# ### Load ROI Timeseries, TR and WinInfo

file_name  = SBJ+'_CTask001_WL0'+str(WL_secs)+'_WS01_NROI0200_dF.mat'             # Data file name
data_path  = osp.join('/data/SFIMJGC_HCP7T/PRJ_CognitiveStateDetection02/PrcsData_PNAS2015',SBJ,'D02_CTask001',file_name) # Path to data
data_df    = loadmat(data_path) # Read data
win_labels = [s[0:4] for s in data_df['winInfo'][0][0][4]]
TR_secs    = data_df['TR'][0][0]

np.savetxt('/data/SFIMJGC_HCP7T/manifold_learning_fmri/Resources/Figure01/winlabels_wl030_ws001.csv',np.array(win_labels),fmt='%s')

# ### Put the Timeseries into a DataFrame

roi_ts         = pd.DataFrame(data_df['origTS'])
roi_ts.columns = ['ROI'+str(i).zfill(3) for i in np.arange(roi_ts.shape[1])]
roi_ts.index   = pd.timedelta_range(start='0',periods=roi_ts.shape[0],freq='{tr}L'.format(tr=TR_secs*1000))
print('++ INFO: Time series dataframe shape: % s' % str(roi_ts.shape))
roi_ts.head(5)

roi_ts.to_csv('/data/SFIMJGC_HCP7T/manifold_learning_fmri/Resources/Figure01/sbj06_ctask001_nroi0200_wl030_ws001.csv')
