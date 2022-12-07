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
#     display_name: Embeddings2 + Sdim
#     language: python
#     name: embeddings3
# ---

# # DESCRIPTION: Figure 2. LE step-by-step
#
# This notebook is used to generate figure 2, which describes the different steps involved in the LE algorithm. 
#
# The algorithm is demonstrated using one run from the multi-task dataset (Gonzalez-Castillo et al. 2015).
#
# The code in this notebook contains portions of the SpectralEmbedding function from scikit-learn that implements the LE algorithm
#
# At the end of this notebook, we also generate the embedding on a single step using the Spectral Embedding function to compare with the step-by-step approach.
#
# All other notebooks rely on using the SpectralEmbedding interface from scikit-learn

# +
import pandas as pd
import numpy as np
import os.path as osp
import os
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import seaborn as sns
import hvplot.networkx as hvnx
import hvplot.pandas 
import plotly.express as px
import panel as pn
from IPython.display import Image
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold._spectral_embedding import _deterministic_vector_sign_flip
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from utils.random import seed_value
from utils.basics import PRJ_DIR, task_cmap_caps
# -

Distance_Function = 'euclidean' # Distance Function
knn               = 90          # Number of Neighbors for the N-Nearest Neighbors Step
drop_XXXX         = False       # Whether or not to include task-inhomogenous windows (e.g., those that contain data across tasks)
n_components      = 3           # Number of dimensions for the final embedding
norm_laplacian    = True        # Whether or not to use the normalized version of the laplacian (Sciki-learn used norm_laplacian = True)

if not osp.exists(osp.join(PRJ_DIR,'Outputs','Figure03')):
    os.makedirs(osp.join(PRJ_DIR,'Outputs','Figure03'))

# ***
# ### 1. Load representative SWC Matrix
#
# First we load a tvFC matrix for a representative run from the multi-task dataset. Entries in this matrix are the Fisher's transform of the Pearson's correlation between windowed ROI timeseries.

swc_path    = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','SBJ06','Original','SBJ06_Craddock_0200.WL045s.WS1.5s.tvFC.Z.asis.pkl')
swc         = pd.read_pickle(swc_path)
win_labels  = swc.columns

print('++ INFO: Size of SWC dataframe is %d connections X %d windows.' % swc.shape)
swc.head(5)

# This code can be run in two ways:
#
# * Using all available sliding windows (```drop_XXXX = False```)
# * Using only task-homogenous windows (```drop_XXXX = True```). We use the term task-homogenous windows to refer to sliding windows that fall completely within the temporal span of a single task, and do not include any instruction period and/or combination of two tasks.

if drop_XXXX:
    print("++ WARNING: Dropping task inhomogenous windows")
    swc        = swc.drop('XXXX',axis=1)
    win_labels = swc.columns
print("++ INFO: Final Number of windows = %d winsows" % swc.shape[1])

# ##### 1.1. Compute location of annotations for the different matrices
# In order to annotate figures, we need to calcuate where task transition occurs and also the midpoint for each block of windows corresponding to each task. We will use those to position labels in upcoming figures and to draw dotted lines. These values are not used by the LE algorithm and are only for visualization purposes.

win_labels = pd.Series(swc.columns)                                       # Task associated with each window
line_idxs = win_labels[(win_labels != win_labels.shift(1))].index.values  # Indexes to windows where task changes (e.g., REST -> XXXX, BACK --> VIDEO)
aux = win_labels[(win_labels != win_labels.shift(1))]
aux = aux[aux!='XXXX']                                                    # Becuase XXXX occurs for very few windows, we will not draw labels for those.
tick_idxs = aux.index + 45                                                # Midpoint for each task block (in terms of windows)
tick_labels = aux.values                                                  # Label for each midpoint.

# ##### 1.2. Plot the tvFC matrix (Panel A)
# Next, we draw the tvFC matrix that constitute the input to the LE algorithm. We annotate the matrix with small colored segments at the top signaling the different task periods. Grey = Rest, Blue = 2-Back, Yellow = Video and Green = Math.

plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(30,5))
mat = sns.heatmap(swc,cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False,cbar_kws={'label': 'tvFC (Z)'})
mat.set_xlabel('Time [Windows]', fontsize=26)
mat.set_ylabel('Connections',    fontsize=26)
for x,l in zip(tick_idxs,tick_labels):
    mat.add_patch(Rectangle((x-45, -301),  91, 300, fill=True, color=task_cmap_caps[l], lw=0, clip_on=False))
cbar = mat.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.label.set_size(26)
cbar.ax.yaxis.set_label_position('left')

# ***
# ### 2. LE STEP 1: Compute Dissimilarity Matrix
#
# The first step of the LE algorithm is the computation of a pair-wise disimilarity metric. Here we will rely on the euclidean distance. We first compute the matrix using scipy pdist function and then place the resulting matrix into a pandas dataframe with meaninful labels

DS = squareform(pdist(swc.T, Distance_Function))
DS = pd.DataFrame(DS,columns=win_labels, index=win_labels)

# ##### 2.1 Plot the Dissimilary Matrix (Panel B)

fig,ax = plt.subplots(1,1,figsize=(11,8))
mat = sns.heatmap(DS, cmap='viridis', ax=ax, vmin=0, vmax=65, square=True,cbar_kws={'label': 'Euclidean Distance'})
for idx in line_idxs:
    ax.plot([idx,idx],[0,DS.shape[0]],'k--')
    ax.plot([0,DS.shape[0]],[idx,idx],'k--')
ax.yaxis.set_ticks(tick_idxs);
ax.xaxis.set_ticks(tick_idxs);
ax.yaxis.set_ticklabels(tick_labels,fontsize=18);
ax.xaxis.set_ticklabels(tick_labels,fontsize=18);
cbar = mat.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.label.set_size(26)
cbar.ax.yaxis.set_label_position('left')

# ***
# ### 3. LE STEP 2: Compute the Affinity Matrix
#
# The second step is the conversion of the disimilary matrix into an affinity matrix using the N-nearest neighbor algorithm. Here for ilustative purposes we will use knn=90.

W = kneighbors_graph(swc.T, knn, include_self=False, n_jobs=-1, metric=Distance_Function, mode='connectivity').toarray()

# Although DS is symmetric, a call to ```kneighbors_graph``` might not necessarily render a symmetric affinity matrix becuase there is no guarantee that if node i is within the knn nearest neighbors of node j, the opposite is also true. 
#
# To ensure that W is symmetric, we have two options:
#     
# * Wij = 1 if and only if i is a neighbor of j **and** j is a neighbor of i.
# * Wij = 1 if and only if i is a neighbor of j **or** j is a neighbor of i.
#
# In the original manuscript by Belkin et al., this step of the LE algorithm is described as follows: "odes i and j are connected by an edge if i is among n nearest neighbors of j or j is among n nearest neighbors of i"
#
# This is what we implement on the next line

# ##### 3.1. Plot the affinity matrix (Panel C)

# Generate a graph view of the W matrix

# ##### 3.2. Generate Graph View of Affinity Matrix
#
# The affinity matrix can also be thought of as a graph. Next we show this alternative view.
#
# First, we create a networkX graph object using W

# Generate a graph layout using the spring function from NX. Any other layout would work too, as layout does not change the graph

# Next, we compute a layout for the graph based on the spring algorithm. The same graph can be represented in many different ways. For other options, please check the layout functions in NetworkX.

# Draw the graph view of W

pos = nx.layout.spring_layout(G, seed=seed_value)

g_plot=hvnx.draw(G,pos,node_color='white', edge_width=0.1, edge_color='purple', node_size=150, node_edge_color='lightgray')
pn.pane.HoloViews(g_plot).save(osp.join(PRJ_DIR,'Outputs','Figure03','G_white.png'))

# Let's now plot the same graph but annotating each node by the task being performed during the window the node represents

# The LE algorithm makes no use of the task information when generating the lower dimensional embedding. Yet, to visualize how the graph captures imporant aspects of the multi-task dataset, below we generate an additional view of the same graph with the same layout, but this time nodes are colored according to task instead of all of them being white color

# Obtain a list of unique labels, so that we can plot nodes corresponding to each task one-by-one with differnet colors
unique_win_labels = win_labels.unique()
unique_win_labels.sort()
unique_win_labels

# This shows a static version of the figure (for github). If running the notebook yourself, simply add g_plot to a new cell
# so you can see and interact with the graph
Image("../Outputs/Figure03/G_colored.png")

# ***
# ### 4. LE STEP 3: Generate the Graph Laplacian
#
# To compute the laplacian matrix we rely on scipy function laplacian. This function can return both the laplacian and also a separate array with node degree informaiton (D).
#
# This function can also return the normalized versions of these two structures (L and D) if normed = True. We used the normalized version for consistency with sikit-learn implementation of the LE algorithm.

L,D = laplacian(W.values,normed=True, return_diag=True)

# The following sign change operation is performed for consistency with the scikit-learn implementation. According to the source code, the rationale for this step is:
#     "We are computing the opposite of the laplacian inplace so as to spare a memory allocation of a possibly very large array"

L *= -1

# ***
# ### 5. LE STEP 4: Eigenvalue Decomposition
#
# First, we generate a vector or random numbers needed by scipy eigsh function, which performs spectral decomposition of a given input matrix.

v0 = np.random.RandomState(seed_value).uniform(-1,1,L.shape[0])

# Becuase the first eigenvector is always discarded (as it is all ones), we always need to keep one more than the requested dimensions. In the next line we increase by one the number of dimensions being requested to accomodate for this.

n_components = n_components + 1
print('++ [LE]: Number of Components = %d' % n_components)

# Finally, we use scipy eigsh function to perform the spectral decomposition of the Laplacian matrix

eigenvalues, eigenvectors = eigsh(L, k=n_components, sigma=1.0, which="LM", tol=0.0, v0=v0)

print('++ Eigenvalues = %s' % str(eigenvalues))

eigenvectors_df = pd.DataFrame(eigenvectors,columns=['f_{}'.format(i) for i in np.arange(n_components)])
eigenvectors_df.columns.name='Eigenvectors'
eigenvectors_df['win_labels'] = win_labels
eigenv_values_figure = eigenvectors_df.hvplot(hover_cols=['win_labels'], title='Eigenvectors', xlabel='Time [Window ID]', width=1000)
pn.pane.HoloViews(eigenv_values_figure).save('../Outputs/Figure03/eigenvalues_plot.png')

# This shows a static version of the figure (for github). If running the notebook yourself, simply add eigenv_values_figure to a new cell
# so you can see and interact with the graph
Image("../Outputs/Figure03/eigenvalues_plot.png")

# ***
# ### 6. LE STEP 5: Compute Embedding
#
# The last step does some final transformations (remove first eigenvalue, account for using the normalized laplacian, ensure same sign) needed to generate the final embeddings

embedding = eigenvectors.T[n_components::-1]

if norm_laplacian:
    # recover u = D^-1/2 x from the eigenvector output x
    embedding = embedding / D

embedding = _deterministic_vector_sign_flip(embedding)

embedding = embedding[1:n_components].T

embedding_df = pd.DataFrame(embedding,columns=['f_{}'.format(i+1) for i in np.arange(n_components-1)])
embedding_df.columns.name='Eigenvectors'
embedding_df['win_labels'] = win_labels
embedding_fig = embedding_df.hvplot(hover_cols=['win_labels'], title='Embedding', xlabel='Time [Window ID]', width=1000)
pn.pane.HoloViews(embedding_fig).save('../Outputs/Figure03/embedding_plot.png')

# This shows a static version of the figure (for github). If running the notebook yourself, simply add embedding_fig to a new cell
# so you can see and interact with the graph
Image("../Outputs/Figure03/embedding_plot.png")

# ***
# ### 7. Select Dimensions and Plot (3D - Panel E)

LE_steps_3D         = pd.DataFrame(100*embedding,columns=['D'+str(i+1).zfill(3) for i in np.arange(3)])
LE_steps_3D['Task'] = win_labels
LE_steps_3D['size'] = 1
LE_steps_3D.head(5)

scene = dict(
        xaxis = dict(nticks=4, range=[-0.5,0.5], gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=4, range=[-0.5,0.5],gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=4, range=[-0.5,0.5],gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))

# +
fig = px.scatter_3d(LE_steps_3D,x='D001',y='D002',z='D003', width=500, height=500, color='Task', size='size', color_discrete_sequence=['gray','pink','blue','yellow','green'], size_max=10)
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1, y=1, z=2)
)
fig.update_layout(scene_camera=camera,scene=scene,scene_aspectmode='cube',margin=dict(l=0, r=0, b=0, t=0))
fig.update_traces(marker=dict(line=dict(width=0)))

fig.write_image('../Outputs/Figure03/embedding_3d.png')
# -

# This shows a static version of the figure (for github). If running the notebook yourself, simply add fig.show() to a new cell
# so you can see and interact with the graph
Image("../Outputs/Figure03/embedding_3d.png")

embedding_2d = LE_steps_3D.hvplot.scatter(x='D001',y='D002',color='Task',cmap=task_cmap_caps,aspect='square', fontsize={'labels':20,'ticks':16})
pn.pane.HoloViews(embedding_2d).save('../Outputs/Figure03/embedding_2d.png')

# This shows a static version of the figure (for github). If running the notebook yourself, simply add embedding_2d to a new cell
# so you can see and interact with the graph
Image("../Outputs/Figure03/embedding_2d.png")
