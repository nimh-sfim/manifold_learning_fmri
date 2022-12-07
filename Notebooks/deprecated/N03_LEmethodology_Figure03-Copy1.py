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
from sklearn.neighbors import kneighbors_graph
from matplotlib.patches import Rectangle
import plotly.express as px
hv.extension('bokeh')

from scipy.spatial.distance import euclidean   as euclidean_distance
from scipy.spatial.distance import cosine      as cosine_distance
from scipy.spatial.distance import correlation as correlation_distance
from scipy.stats import pearsonr as correlation

from utils.basics import PRJ_DIR, task_cmap

dmetric_list = ['euclidean','correlation','cosine']
dmetric_mins = [0,0,0]
dmetric_maxs = [60,.8,.8]

# ### Load SWC Matrix

swc_path    = osp.join(PRJ_DIR,'Resources','Figure03','swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv')
swc         = pd.read_csv(swc_path, index_col=[0,1], header=0)
# Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
swc.columns = swc.columns.str.split('.').str[0]

print('++ INFO: Size of SWC dataframe is %d connections X %d windows.' % swc.shape)
swc.head(5)

fig = plt.figure(figsize=(40,5))
sns.heatmap(swc, cmap='RdBu_r', vmin=-0.75, vmax=0.75)

# ### Drop In-between windows

swc        = swc.drop('XXXX',axis=1)
win_labels = swc.columns

plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(40,5))
mat = sns.heatmap(swc,cmap='RdBu_r', vmin=-0.75, vmax=0.75, xticklabels=False, yticklabels=False)
mat.set_xlabel('Time [Windows]', fontsize=26)
mat.set_ylabel('Connections',    fontsize=26)
mat.add_patch(Rectangle((0, -301),  91, 300, fill=True, color=task_cmap['Rest'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91, -301), 91, 300, fill=True, color=task_cmap['Memory'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91*2, -301), 91, 300, fill=True, color=task_cmap['Video'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91*3, -301), 91, 300, fill=True, color=task_cmap['Math'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91*4, -301), 91, 300, fill=True, color=task_cmap['Memory'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91*5, -301), 91, 300, fill=True, color=task_cmap['Rest'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91*6, -301), 91, 300, fill=True, color=task_cmap['Math'], lw=0, clip_on=False))
mat.add_patch(Rectangle((91*7, -301), 91, 300, fill=True, color=task_cmap['Video'], lw=0, clip_on=False))

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
    axs[idx].yaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO'],fontsize=18);
    axs[idx].xaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO'],fontsize=18);
    for x in [0,91,182,273,364,455,546,637,dist_matrix[dm].shape[0]]:
        axs[idx].plot([x,x],[0,dist_matrix[dm].shape[0]],'k--')
        axs[idx].plot([0,dist_matrix[dm].shape[0]],[x,x],'k--')

aff_matrix = {}
for dmetric in dmetric_list:
    aff_matrix[dmetric] = pd.DataFrame(kneighbors_graph(swc.T, 90, include_self=False, n_jobs=-1, metric=dmetric, mode='connectivity').toarray())
    #From Belkin: "Nodes i and j are connected by an edge if i is among n nearest neighbors of j or j is among n nearest neighbors of i"
    # The above aff_matrix[dmetric] is not necessarily symetric. After the following transformation that takes into account the statement in the
    # previous line, it will be.
    aff_matrix[dmetric] = ((0.5 * (aff_matrix[dmetric] + aff_matrix[dmetric].T)) > 0).astype(int) 

fig,axs = plt.subplots(1,3,figsize=(24,5))
for idx,dm in enumerate(dmetric_list):
    sns.heatmap(aff_matrix[dm], cmap='viridis', vmin=0, vmax=1, ax=axs[idx], square=True)
    axs[idx].yaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
    axs[idx].xaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
    axs[idx].yaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO'],fontsize=18);
    axs[idx].xaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO'],fontsize=18);
    for x in [0,91,182,273,364,455,546,637,dist_matrix[dm].shape[0]]:
        axs[idx].plot([x,x],[0,dist_matrix[dm].shape[0]],'w--')
        axs[idx].plot([0,dist_matrix[dm].shape[0]],[x,x],'w--')

import networkx as nx
import hvplot.networkx as hvnx

unw_graphs, unw_graph_layouts = {}, {}
for dmetric in dmetric_list:
    unw_graphs[dmetric] = nx.from_numpy_matrix(aff_matrix[dmetric].values)
    unw_graph_layouts[dmetric] = nx.layout.spring_layout(unw_graphs[dmetric])

graph_figs = {}
for dmetric in dmetric_list:
    graph_figs[dmetric] = hvnx.draw(unw_graphs[dmetric],unw_graph_layouts[dmetric] ,node_color='purple', edge_width=0.5, edge_color='black') + (hvnx.draw(unw_graphs[dmetric],unw_graph_layouts[dmetric]) * \
                          hvnx.draw(unw_graphs[dmetric],unw_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'REST')[0].tolist(), node_color=task_cmap['Rest'], edge_width=0.5) * \
                          hvnx.draw(unw_graphs[dmetric],unw_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'VIDE')[0].tolist(), node_color=task_cmap['Video'], edge_width=0.5) * \
                          hvnx.draw(unw_graphs[dmetric],unw_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'MATH')[0].tolist(), node_color=task_cmap['Math'], edge_width=0.5) * \
                          hvnx.draw(unw_graphs[dmetric],unw_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'BACK')[0].tolist(), node_color=task_cmap['Memory'], edge_width=0.5)) 

(graph_figs['euclidean'] + graph_figs['correlation'] + graph_figs['cosine']).cols(6)

from networkx import laplacian_matrix, normalized_laplacian_matrix, degree_mixing_matrix

L = laplacian_matrix(unw_graphs['euclidean']).toarray()

pd.DataFrame(L, columns=['W'+str(i).zfill(3) for i in np.arange(728)], index=['W'+str(i).zfill(3) for i in np.arange(728)])

fig,ax = plt.subplots(1,1,figsize=(20,20))
L = laplacian_matrix(unw_graphs['euclidean']).toarray()
sns.heatmap(L, cmap='viridis', ax=ax, square=True)
axs[idx].yaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
axs[idx].xaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
axs[idx].yaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO'],fontsize=18);
axs[idx].xaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO'],fontsize=18);
for x in [0,91,182,273,364,455,546,637,dist_matrix[dm].shape[0]]:
    axs[idx].plot([x,x],[0,dist_matrix[dm].shape[0]],'w--')
    axs[idx].plot([0,dist_matrix[dm].shape[0]],[x,x],'w--')

aff_matrix_w = {}
for dmetric in dmetric_list:
    aff_matrix_w[dmetric] = pd.DataFrame(np.exp(-dist_matrix[dmetric]/10) * aff_matrix[dmetric].values)

fig,axs = plt.subplots(1,3,figsize=(24,5))
for idx,dm in enumerate(dmetric_list):
    sns.heatmap(aff_matrix_w[dm], cmap='viridis', ax=axs[idx], square=True)
    axs[idx].yaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
    axs[idx].xaxis.set_ticks([45, 136, 227, 318, 409, 500, 591, 682 ])
    axs[idx].yaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO']);
    axs[idx].xaxis.set_ticklabels(['REST','BACK','VIDEO','MATH','BACK','REST','MATH','VIDEO']);

w_graphs, w_graph_layouts = {}, {}
for dmetric in dmetric_list:
    w_graphs[dmetric] = nx.from_numpy_matrix(aff_matrix_w[dmetric].values)
    w_graph_layouts[dmetric] = nx.layout.spring_layout(w_graphs[dmetric])

graph_figs = {}
for dmetric in dmetric_list:
    graph_figs[dmetric] = hvnx.draw(w_graphs[dmetric],w_graph_layouts[dmetric] ,node_color='purple', edge_width=0.5, edge_color='black') + (hvnx.draw(unw_graphs[dmetric],w_graph_layouts[dmetric]) * \
                          hvnx.draw(w_graphs[dmetric],w_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'REST')[0].tolist(), node_color=task_cmap['Rest'], edge_width=0.5) * \
                          hvnx.draw(w_graphs[dmetric],w_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'VIDE')[0].tolist(), node_color=task_cmap['Video'], edge_width=0.5) * \
                          hvnx.draw(w_graphs[dmetric],w_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'MATH')[0].tolist(), node_color=task_cmap['Math'], edge_width=0.5) * \
                          hvnx.draw(w_graphs[dmetric],w_graph_layouts[dmetric] ,nodelist=np.where(win_labels == 'BACK')[0].tolist(), node_color=task_cmap['Memory'], edge_width=0.5)) 

(graph_figs['euclidean'] + graph_figs['correlation'] + graph_figs['cosine']).cols(6)

from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh

# + [markdown] tags=[]
# #### 5.5. Eigen vector decomposition of the normalized Laplacian
#
# The code below has been copied from sklearn implementation of laplacian embeddings. That way we have access to every single step. For example here we could examine the eigenvalues if we were interested in doing so
# -

from scipy import sparse
def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.
    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.
    value : float
        The value of the diagonal.
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.
    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.
    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.
    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.
    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


# Default values in sklearn that are needed
random_state = 43
drop_first = True
norm_laplacian = True
n_components = 3
eigen_tol = 0.0

if drop_first:
    n_components = n_components + 1
print('++ [LE]: Number of Components = %d' % n_components)

w_laplacians, uw_laplacians = {},{}
uw_eigenvalues, uw_eigenvectors = {},{}
w_eigenvalues, w_eigenvectors = {},{}
w_embedding, uw_embedding = {},{}
w_LE, uw_LE = {},{}
for dmetric in dmetric_list:
    uw_laplacians[dmetric,'L'], uw_laplacians[dmetric,'dd'] = csgraph_laplacian(aff_matrix[dmetric].values, normed=True, return_diag=True)
    w_laplacians[dmetric,'L'],  w_laplacians[dmetric,'dd']  = csgraph_laplacian(aff_matrix_w[dmetric].values, normed=True, return_diag=True)
    
    uw_laplacians[dmetric,'L'] = _set_diag(uw_laplacians[dmetric,'L'], 1, norm_laplacian)
    w_laplacians[dmetric,'L']  = _set_diag(w_laplacians[dmetric,'L'], 1, norm_laplacian)
    
    # We are computing the opposite of the laplacian inplace so as
    # to spare a memory allocation of a possibly very large array
    uw_laplacians[dmetric,'L'] *= -1
    w_laplacians[dmetric,'L']  *= -1
    
    uw_v0 = np.random.RandomState(43).uniform(-1,1,uw_laplacians[dmetric,'L'].shape[0])
    w_v0  = np.random.RandomState(43).uniform(-1,1,w_laplacians[dmetric,'L'].shape[0])

    uw_eigenvalues[dmetric], uw_eigenvectors[dmetric] = eigsh(uw_laplacians[dmetric,'L'], k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=uw_v0)
    w_eigenvalues[dmetric],  w_eigenvectors[dmetric]  = eigsh(w_laplacians[dmetric,'L'],  k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=w_v0)

    uw_embedding[dmetric] = uw_eigenvectors[dmetric].T[n_components::-1]
    w_embedding[dmetric]  = w_eigenvectors[dmetric].T[n_components::-1]
    
    if norm_laplacian:
        # recover u = D^-1/2 x from the eigenvector output x
        uw_embedding[dmetric] = uw_embedding[dmetric] / uw_laplacians[dmetric,'dd']
        w_embedding[dmetric]  = w_embedding[dmetric]  / w_laplacians[dmetric,'dd']
        
    uw_embedding[dmetric] = _deterministic_vector_sign_flip(uw_embedding[dmetric])
    w_embedding[dmetric]  = _deterministic_vector_sign_flip(w_embedding[dmetric])
    
    if drop_first:
        uw_embedding[dmetric] = uw_embedding[dmetric][1:n_components].T
        w_embedding[dmetric]  = w_embedding[dmetric][1:n_components].T

    else:
        uw_embedding[dmetric] = uw_embedding[dmetric][:n_components].T
        w_embedding[dmetric]  = w_embedding[dmetric][:n_components].T
    
    uw_LE[dmetric] = pd.DataFrame(uw_embedding[dmetric],columns=['D'+str(i+1).zfill(3) for i in np.arange(3)])
    uw_LE[dmetric]['Task'] = win_labels
    w_LE[dmetric]  = pd.DataFrame(w_embedding[dmetric],columns=['D'+str(i+1).zfill(3) for i in np.arange(3)])
    w_LE[dmetric]['Task'] = win_labels

fig = px.scatter_3d(uw_LE['cosine'],x='D001',y='D002',z='D003', width=600, height=600, color='Task', color_discrete_sequence=['gray','blue','green','yellow'])
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1, y=3, z=1)
)
fig.update_layout(scene_camera=camera)
fig.show()


