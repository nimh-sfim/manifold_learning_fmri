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

# # Figure 4 - TSNE Step-by-step
#
# This notebook is used to generate Figure 4 and contains a basic implementaiton of the T-SNE algorithm. This notebook contains code available at Laurens Van der Maaten T-SNE home page (https://lvdmaaten.github.io/tsne/), that provides an implementaiton of the T-SNE algorithm as originally described in the 2004 paper.
#
# For this demonstration we will use the tvFC from one run of the multi-task dataset previously published in [Gonzalez-Castillo et al. PNAS (2015)](https://www.pnas.org/doi/abs/10.1073/pnas.1501242112)

# +
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
from utils.plotting import plot_matrix
import panel as pn
import holoviews as hv
hv.extension('bokeh')

from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from utils.basics import task_cmap_caps as cmap

from tqdm.notebook import tqdm

from scipy.ndimage.interpolation import shift
from sklearn.manifold._t_sne import _joint_probabilities
# -

# To ensure reproducibility across executions, we set the random seed in numpy to 43

np.random.seed(43)

# The following cell contains a series of hyper-parameters that affect the output of the T-SNE algorithm. All values set here corresponds to those in the original implementation of the T-SNE algorithm, with the exception of the perplexity which is set to 100 here.

# +
# T-SNE Setup Hyper-paramters
desired_dimensions      = 2           # Final Number of dimensions
desired_perplexity      = 100.0       # Perplexity: related to neighborhood size or a trade-off between focus on local vs. global structure
distance_function       = 'euclidean' # Distance Function

# T-SNE Gradient Descent Hyper-parameters
n_iter                  = 1000        # Maximum Number of Iterations for Gradient Descent (KL Optimization)
initial_momentum        = 0.5         # Initial momentum      
final_momentum          = 0.8         # Final momentum
learning_rate           = 500         # Learning rate
min_gain                = 0.01        #
early_exaggeration      = 4.          # Early Exaggeration Factor
early_exaggeration_ends = 100         # Number of itrations for early exasggeration


# Tolerance-related parameters
tsne_initialization     = 'random' #  'random' or 'pca'
# -

# ***
# # Load tvFC Data
#
# First we load an exemplary tvFC matrix. This will be the same data we used for the description of the LE algorithm on Notebook 3.

# +
print('++ INFO: Loading the tvFC dataset.....')
X_df = pd.read_csv('../Resources/Figure03/swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv.gz', index_col=[0,1])

# Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
X_df.columns = X_df.columns.str.split('.').str[0]
# Extract Task Lbaels (for coloring purposes)
labels  = pd.Series(X_df.columns)
X       = X_df.values.T
X_orig  = X.copy()
(n_wins, n_conns)  = X.shape         # n = number of samples | d = original number of dimensions
print(' +       Input data shape = [%d Windows X %d Connections]' % (n_wins, n_conns))
# -

# To annotate affinity and dissimilarity matrixes, we will next extract the location of transitions between tasks and also the center window of each homogenous task period. The transitions will be used to draw dashed lines and the centers for placement of labels on the axes.

# +
aux       = labels[(labels != labels.shift(1))] # Check for discontinuities in label names.

line_idxs   = aux.index.values  # Indexes to windows where task changes (e.g., REST -> XXXX, BACK --> VIDEO)
tick_idxs   = aux.index + np.diff(list(aux.index) + [n_wins])/2 # Midpoint for each task block (in terms of windows)
tick_labels = aux.values
print('++ INFO: Number of Tick Labels = %d | %s' % (len(tick_labels),str(tick_labels)))
# -

# ***
# # **T-SNE STEP 1.** Compute Affinity Matrix in Original Space

# **Compute Dissimilarity Matrix** First we compute a dissimuilarity matrix with all pair-wise differences between columns of the tvFC matrix. For this we use the Euclidean distance, as that was the distance function used in the original description of the T-SNE algorithm.

# First we compute the square of the euclidean distance (part of the exponential in the conditional probability p)
DS = squareform(pdist(X, distance_function))
print("++ INFO: Shape of dissimilary matrix: %s" % str(DS.shape))

plot_matrix(DS, tick_idxs=tick_idxs, tick_labels=tick_labels, line_idxs=line_idxs)

# **Transform DS into an affinity matrix (P)** 
#
# For this we will use a Gaussian kernel centered on each sample. The variance of the kernel will be point dependent and calculated based on the desired perplexity.

distances = (DS**2).astype(np.float32, copy=True)
P = squareform(_joint_probabilities(distances, desired_perplexity=desired_perplexity,verbose=2))
P = np.maximum(P, 1e-12) # To avoid divisions by zero
print(P.shape)

plot_matrix(P, q_min=0.05, q_max=0.90, ctitle='$P_{j|i}$',tick_idxs=tick_idxs, tick_labels=tick_labels, line_idxs=line_idxs)

# ***
# # TSNE STEP 2. Random initalization of lower dimensional mapping.
#
# This code allows the use of either random or PCA initialization. For the purpose of Figure 4 we used random initialization as that is how the T-SNE algorithm was originally described. If interested in observing how PCA initialization affects the gradient descent portion of the algorithm, change the value of the ```tsne_initialization``` variable to pca and run again.

print('++ INFO: Initialization Mode: %s' % tsne_initialization)
np.random.seed(43)
if tsne_initialization == 'random':
    Y      = np.random.randn(n_wins, desired_dimensions)
    print(Y[0:3,:])
if tsne_initialization == 'pca':
    pca_init = PCA(n_components=desired_dimensions, svd_solver='full')
    Y   = pca_init.fit_transform(X_orig)

# Next we show the dissimilarity matrix associated with the initial random mapping

plot = pd.DataFrame(Y,columns=['x','y']).plot.scatter(x='x',y='y',c='k', figsize=(6,6), s=10)
plot.set_aspect('equal')

# Same as above but using hvplot
pd.DataFrame(Y,columns=['x','y']).hvplot.scatter(x='x',y='y', aspect='square', c='k', fontsize={'labels':16,'ticks':16})

# # TSNE STEP 3. Compute dissimilarity and affinity matrix for initial low dimensional set of points
#
# Dissimilarity is once again computed using the Euclidean Distance.

DS_low = squareform(pdist(Y, 'euclidean')) 
plot_matrix(DS_low,tick_idxs=tick_idxs, tick_labels=tick_labels, line_idxs=line_idxs)

# Next, to go from dissimilarity to affinity matrix in the lower dimensional space, this time we use a T-student kernel instead of the Gaussian kernel

# +
Q_num = 1/ (1 + np.square(DS_low))
np.fill_diagonal(Q_num,0.)
Q_den = np.sum(Q_num)
Q     = Q_num / Q_den
Q = np.maximum(Q, 1e-12) # To avoid divisions by zero
print(Q.shape)

plot_matrix(Q, q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$',tick_idxs=tick_idxs, tick_labels=tick_labels, line_idxs=line_idxs)

# + [markdown] tags=[]
# ***
# # TNSE STEP 3. Optimization via gradient descent
#
# The code on the next cell has been adapted from the implementation of the T-SNE method available at [Laurens van der Maaten website](https://lvdmaaten.github.io/tsne/code/tsne_python.zip)

# +
# Create Initial empty variables
Y_dict    = {} # This dictionary will contain the embedings at each iteration of the gradient descent
Y_dict[0] = Y  # We load the initial embeddings (the one randomly created above)
cost      = [] # This list will contain the cost-function value at the end of each gradient descent iteration
Q_dict    = {} # This dictionary will contain the affinity matrix for the embedding at the end of each gradient descent iteration

dY = np.zeros((n_wins, desired_dimensions))    # This variable will keep the gradient
iY = np.zeros((n_wins, desired_dimensions))    # gradient update
gains = np.ones((n_wins, desired_dimensions))  # and gains
# -

# The initial iterations will run with early exaggeration, which consist on multiplying the affinity matrix P by an early exaggeration factor. 
#
# Later, once we reach the maximum number of early exaggeration iterations, we will remove that multiplicative factor

# Early Exaggeration
P = P * early_exaggeration

for gs_iter in tqdm(range(n_iter)):
    # Using Eq.4 from the Van Maaten paper
    D_low = squareform(pdist(Y, distance_function))
    Q_num = 1/ (1 + np.square(D_low))
    np.fill_diagonal(Q_num,0.)
    Q_den = np.sum(Q_num)
    Q     = Q_num / Q_den
    Q = np.maximum(Q, 1e-12) # To avoid divisions by zero
    Q_dict[gs_iter] = Q
    
    # Compute gradient (Eq. 5)
    PQ = P - Q
    for i in range(n_wins):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * Q_num[:, i], (desired_dimensions, 1)).T * (Y[i, :] - Y), 0)
    
    # Perform the update
    if i < 20:
        momentum = initial_momentum
    else:
        momentum = final_momentum
    gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
            (gains * 0.8) * ((dY > 0.) == (iY > 0.))
    gains[gains < min_gain] = min_gain
    iY = momentum * iY - learning_rate * (gains * dY)
    Y = Y + iY
    Y = Y - np.tile(np.mean(Y, 0), (n_wins, 1))
    Y_dict[gs_iter+1] = Y
    
    # Compute current value of cost function
    C = np.sum(P * np.log(P / Q))
    cost.append(C)
    
    # End of early exaggeration (in the paper it says to stop after 50 iterations, here it is after 100)
    if gs_iter == early_exaggeration_ends:
        P = P / early_exaggeration

# ***
# # Show Results
#
# First, we look at the evolution of the cost function with gradient descent iterations

gd_obj_df = pd.Series(cost)
gd_obj_df.index.name = 'Iteration'
gd_obj_df.name = 'K-L Divergence'
gd_obj_df.hvplot(c='k', width=1000, fontsize={'labels':16,'ticks':14, 'title':18}) * hv.VLine(early_exaggeration_ends).opts(line_width=0.75, line_color='purple', line_dash='dashed')

gd_obj_df.plot(figsize=(20,5),c='k', title='K-L Divergence', lw=3)
plt.plot([early_exaggeration_ends,early_exaggeration_ends],[0,16],'m--');

# Next, we plot the final embedding generated at the end of the last gradient descent iteration

tsne_result_df = pd.DataFrame(Y_dict[n_iter],columns=['x','y'])
tsne_result_df['labels'] = labels
tsne_result_df['colors'] = [cmap[l] for l in labels]
plot = tsne_result_df.plot.scatter(x='x',y='y',c='colors', figsize=(6,6), s=10)
plot.set_aspect('equal')

# Finally, to explore the evolution of the embedding as gradient descent iterations increase, we create a dynamic dashboard

# %%time
Q_images = {}
for i in tqdm(np.arange(n_iter)):
    Q_images[i] =  plot_matrix(Q_dict[i],q_min=0.05, q_max=0.90, ctitle='',figsize=(7,6),lab_fsize=10)

# +
gs_player = pn.widgets.Player(name='Gradient Descent', start=0, end=n_iter-1, value=0, width=1000)
@pn.depends(gs_player)
def plot_objective(i):
    obj_curve     = gd_obj_df.hvplot(c='k', width=800, fontsize={'labels':16,'ticks':14, 'title':18}, title='Iternation Number = %d' % i)
    end_ee_marker = hv.VLine(early_exaggeration_ends).opts(line_width=0.75, line_color='purple', line_dash='dashed')
    iter_marker   = hv.Points((i,gd_obj_df[i])).opts(size=10, color='r', marker='s')
    return (obj_curve * end_ee_marker * iter_marker).opts(toolbar=None)

@pn.depends(gs_player)
def plot_embed(gd_iter):
    df = pd.DataFrame(Y_dict[gd_iter], columns=['x','y'])
    df['label'] = labels
    scat_plot = df.hvplot.scatter(x='x',y='y',color='label',aspect='square', cmap=cmap, fontsize={'labels':18,'ticks':18, 'title':18}, frame_width=375, legend=False).opts(legend_position='top')
    return scat_plot.opts(toolbar=None)

@pn.depends(gs_player)
def plot_Q(gd_iter):
    return Q_images[gd_iter]


# -

dashboard = pn.Column(gs_player,plot_objective,pn.Row(plot_embed, plot_Q))

port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

dashboard_server = dashboard.show(port=port_tunnel,open=False)

# +
#dashboard_server.stop()
