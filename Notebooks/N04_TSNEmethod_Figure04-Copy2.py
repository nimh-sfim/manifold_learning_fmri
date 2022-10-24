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
# This notebook contains code available at Laurens Van der Maaten T-SNE home page (https://lvdmaaten.github.io/tsne/) 

# +
import numpy as np
import pandas as pd
import pylab
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
from scipy.spatial.distance import pdist, squareform
from utils.tsne_functions import get_perplexity, get_Pij_row, gauss_var_search, get_P
from utils.basics import task_cmap_caps
from tqdm.notebook import tqdm
import panel as pn
import holoviews as hv
hv.extension('bokeh')
np.random.seed(43)

from scipy.ndimage.interpolation import shift
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import squareform

# +
# T-SNE Setup Hyper-paramters
n_components            = 2          # Final Number of dimensions
desired_perplexity      = 100.0      # Neighborhood size

# T-SNE Optimization Hypter-parameters
n_iter                  = 1000        # Maximum Number of Iterations for Gradient Descent (KL Optimization)
initial_momentum        = 0.5        # Those match the values in sklearn t-sne implementation according to the comments      
final_momentum          = 0.8        # 
learning_rate           = 100 #500
min_gain                = 0.01
early_exaggeration      = 4.
early_exaggeration_ends = 50 


# Tolerance-related parameters
perp_tol                = 1e-5 # Tolerance for differences np.log(perplexity) - SAME AS SKLEARN
perp_n_attempts         = 50
tsne_initialization     = 'pca' #  'random' or 'pca'


# -

def plot_matrix(m,q_min=0.01, q_max=0.99, cmap='viridis', ctitle='Euclidean Distance', figsize=(11,8), lab_fsize=18):
    fig,ax = plt.subplots(1,1,figsize=figsize);
    mat = sns.heatmap(m, vmin=np.quantile(m,q_min), vmax=np.quantile(m,q_max), square=True, cmap=cmap,cbar_kws={'label': ctitle});
    for idx in line_idxs:
        ax.plot([idx,idx],[0,m.shape[0]],'w--');
        ax.plot([0,m.shape[0]],[idx,idx],'w--');
    ax.yaxis.set_ticks(tick_idxs);
    ax.xaxis.set_ticks(tick_idxs);
    ax.yaxis.set_ticklabels(tick_labels,fontsize=lab_fsize);
    ax.xaxis.set_ticklabels(tick_labels,fontsize=lab_fsize);
    cbar = mat.collections[0].colorbar;
    cbar.ax.tick_params(labelsize=16);
    cbar.ax.yaxis.label.set_size(20);
    cbar.ax.yaxis.set_label_position('left');
    plt.close()
    return fig


# # 1. Load tvFC Data

print('++ INFO: Loading the tvFC dataset.....')
X_df = pd.read_csv('../Resources/Figure03/swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv.gz', index_col=[0,1])
# Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
X_df.columns = X_df.columns.str.split('.').str[0]
labels  = pd.Series(X_df.columns)
X       = X_df.values.T
X_orig  = X.copy()
cmap    = task_cmap_caps
print(' +       Input data shaep = %s' % str(X.shape))

n,_ = X.shape
w_onsets  = np.arange(0,n)
w_offsets = w_onsets+30
aux = pd.DataFrame(np.zeros((n,1017)))
aux.columns.name = 'TR'
aux.index.name = 'Window'
for w in np.arange(n):
    aux.loc[w,w_onsets[w]:w_offsets[w]-1] = 1
aux.head(5)

aux_b = pd.DataFrame(np.zeros((n,n)))
aux_b.index.name='Window_x'
aux_b.columns.name='Window_y'
for x in tqdm(np.arange(w)):
    for y in np.arange(w):
        aux_b.loc[x,y]=(aux.loc[x,:] * aux.loc[y,:]).sum()
aux_b

n = X.shape[0]
line_idxs = labels[(labels != labels.shift(1))].index.values  # Indexes to windows where task changes (e.g., REST -> XXXX, BACK --> VIDEO)
aux = labels[(labels != labels.shift(1))]
#aux = aux[aux!='XXXX']                                                    # Becuase XXXX occurs for very few windows, we will not draw labels for those.
tick_idxs = aux.index + np.diff(list(aux.index) + [n])/2 # Midpoint for each task block (in terms of windows)
tick_labels = aux.values
print('++ INFO: Number of Tick Labels = %d | %s' % (len(tick_labels),str(tick_labels)))

# ***
# # 2. T-SNE STEP 1. Compute Similarities in terms of the conditional probabilities of a Gaussian

# #### 1.1. Compute Dissimilarity Matrix

# First we compute the square of the euclidean distance (part of the exponential in the conditional probability p)
DS = squareform(pdist(X, 'euclidean'))
print("++ INFO: Shape of dissimilary matrix: %s" % str(DS.shape))

plot_matrix(DS)

pd.DataFrame(squareform(DS)).hvplot.hist(normed=True) * \
pd.DataFrame(squareform(DS)).hvplot.kde(title='Distribution of distances in original space', xlabel='Pair-wise Euclidean Distances', ylabel='Density', fontsize={'labels':16, 'ticks':14})

a = pd.DataFrame([squareform(DS), squareform(aux_b, checks=False)]).T
a.columns = ['D','Overlap']

a.plot.scatter(x='Overlap',y='D', s=1)
xdata = a['Overlap'].values
ydata = a['D'].values
z = np.polyfit(xdata,ydata,2)
x =np.linspace(1e-12,50,51)
plt.plot(x,np.polyval(z,x))

a['D_corrected'] = a['D'] - np.polyval(z,a['D'])
a['D_corrected'] = a['D_corrected'] - a['D_corrected'].min()

a.plot.scatter(x='Overlap',y='D_corrected', s=1)

DS=squareform(a['D_corrected'])

x =np.linspace(1e-12,50,51)
def sigmoid(x):
        return 1 / (1 + np.exp(-x))
def log_growth(x,a,b,c):
    return c / (1 + a*np.exp(-b*x))
def log_decay(x,a,b,c):
    return c - log_growth(x,a,b,c)
def log_decay(x,a,b):
    return 30 - log_growth(x,a,b,30)
def line(x,a,b):
    return a*x + b


xdata = a['Overlap'].values
ydata = a['D'].values
popt, pcov = curve_fit(line,xdata, ydata)
popt

plt.plot(xdata, ydata,'o',c='gray',markersize=1)
plt.plot(x,log_decay(x, *popt), 'g--' )

modelname.fit(xtrain, ytrain)
prediction = modelname.predict(x_test)
residual = (y_test - prediction)

# + active=""
# ## Attempt at normalizing the distribution
# ## =======================================
# from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
# qt_tr = QuantileTransformer(output_distribution='normal')
# mm_sc = MinMaxScaler()
# DS_n = squareform(np.squeeze(mm_sc.fit_transform(qt_tr.fit_transform(squareform(DS).reshape(-1,1)))))
# plot_matrix(DS_n)

# + active=""
# pd.DataFrame(squareform(DS_n)).hvplot.hist(normed=True) * \
# pd.DataFrame(squareform(DS_n)).hvplot.kde(title='Distribution of distances in original space', xlabel='Pair-wise Euclidean Distances', ylabel='Density', fontsize={'labels':16, 'ticks':14})
# -

# DS = DS_n

# #### 1.2. Transform DS into an affinity matrix given the perplexityCompute Dissimilarity Matrix

distances = (DS**2).astype(np.float32, copy=True)
P = squareform(_joint_probabilities(distances, desired_perplexity=desired_perplexity,verbose=2))
P = np.maximum(P, 1e-12) # To avoid divisions by zero
print(P.shape)

plot_matrix(P, q_min=0.05, q_max=0.90, ctitle='$P_{j|i}$')

(n, d) = X.shape         # n = number of samples | d = original number of dimensions
print('++ INFO: Number of samples = %d | Number of dimensions = %d' % (n,d))
print('++ INFO: Initialization Mode: %s' % tsne_initialization)
np.random.seed(43)
if tsne_initialization == 'random':
    Y      = np.random.randn(n, n_components)
    print(Y[0:3,:])
if tsne_initialization == 'pca':
    pca_init = PCA(n_components=n_components, svd_solver='full')
    Y   = pca_init.fit_transform(X_orig)
dY     = np.zeros((n, n_components))
iY     = np.zeros((n, n_components))
gains  = np.ones((n, n_components))

pd.DataFrame(Y,columns=['x','y']).hvplot.scatter(x='x',y='y', aspect='square', c='k', fontsize={'labels':16,'ticks':16})

# ***
# ## 2. Create a initial random points in low dimensional space

(n, d) = X.shape         # n = number of samples | d = original number of dimensions
print('++ INFO: Number of samples = %d | Number of dimensions = %d' % (n,d))
print('++ INFO: Initialization Mode: %s' % tsne_initialization)
np.random.seed(43)
if tsne_initialization == 'random':
    Y      = np.random.randn(n, n_components)
    print(Y[0:3,:])
if tsne_initialization == 'pca':
    pca_init = PCA(n_components=n_components, svd_solver='full')
    Y   = pca_init.fit_transform(X_orig)
dY     = np.zeros((n, n_components))
iY     = np.zeros((n, n_components))
gains  = np.ones((n, n_components))

pd.DataFrame(Y,columns=['x','y']).hvplot.scatter(x='x',y='y', aspect='square', c='k', fontsize={'labels':16,'ticks':16})

# ## 2.1. Compute affinity matrix for initial low dimensional set of points

DS_low = squareform(pdist(Y, 'euclidean'))
print(DS_low.shape)

plot_matrix(DS_low)

Q_num = 1/ (1 + np.square(DS_low))
np.fill_diagonal(Q_num,0.)
Q_den = np.sum(Q_num)
Q     = Q_num / Q_den
Q = np.maximum(Q, 1e-12) # To avoid divisions by zero
print(Q.shape)

plot_matrix(Q, q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$')

# +
# Run iterations
Y_dict    = {}
Y_dict[0] = Y
cost      = []
Q_dict    = {}

# Early Exaggeration
P = P * early_exaggeration

for gs_iter in tqdm(range(n_iter)):
    # Using Eq.4 from the Van Maaten paper
    D_low = squareform(pdist(Y, 'euclidean'))
    Q_num = 1/ (1 + np.square(D_low))
    np.fill_diagonal(Q_num,0.)
    Q_den = np.sum(Q_num)
    Q     = Q_num / Q_den
    Q = np.maximum(Q, 1e-12) # To avoid divisions by zero
    Q_dict[gs_iter] = Q
    
    # Compute gradient (Eq. 5)
    PQ = P - Q
    for i in range(n):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * Q_num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)
    
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
    Y = Y - np.tile(np.mean(Y, 0), (n, 1))
    Y_dict[gs_iter+1] = Y
    
    # Compute current value of cost function
    C = np.sum(P * np.log(P / Q))
    cost.append(C)
    
    # End of early exaggeration (in the paper it says to stop after 50 iterations, here it is after 100)
    if gs_iter == early_exaggeration_ends:
        P = P / early_exaggeration
# -

gd_obj_df = pd.Series(cost)
gd_obj_df.index.name = 'Iteration'
gd_obj_df.name = 'K-L Divergence'

Q_images = {}
for i in tqdm(np.arange(n_iter)):
    Q_images[i] =  plot_matrix(Q_dict[i],q_min=0.05, q_max=0.90, ctitle='',figsize=(7,6),lab_fsize=10)

# +
gs_player = pn.widgets.Player(name='Gradient Descent', start=0, end=n_iter, value=0, width=1000)
@pn.depends(gs_player)
def plot_objective(i):
    obj_curve     = gd_obj_df.hvplot(c='k', width=800, fontsize={'labels':16,'ticks':14, 'title':18})
    end_ee_marker = hv.VLine(early_exaggeration_ends).opts(line_width=0.75, line_color='purple', line_dash='dashed')
    iter_marker   = hv.Points((i,gd_obj_df[i])).opts(size=10, color='r', marker='s')
    return (obj_curve * end_ee_marker * iter_marker).opts(toolbar=None)

@pn.depends(gs_player)
def plot_embed(gd_iter):
    df = pd.DataFrame(Y_dict[gd_iter], columns=['x','y'])
    df['label'] = labels
    scat_plot = df.hvplot.scatter(x='x',y='y',color='label',aspect='square', cmap=cmap, fontsize={'labels':16,'ticks':14, 'title':18}, frame_width=400).opts(legend_position='top')
    return scat_plot.opts(toolbar=None)

@pn.depends(gs_player)
def plot_Q(gd_iter):
    return Q_images[gd_iter]


# -

pn.Column(gs_player,plot_objective,pn.Row(plot_embed, plot_Q))







gd_obj_df = pd.Series(cost)
gd_obj_df.index.name = 'Iteration'
gd_obj_df.name = 'K-L Divergence'
gd_obj_df.hvplot(c='k', width=1000, fontsize={'labels':16,'ticks':14, 'title':18}) * hv.VLine(early_exaggeration_ends).opts(line_width=0.75, line_color='purple', line_dash='dashed')


# +
def return_map_for_iter(gd_iter):
    df = pd.DataFrame(Y_dict[gd_iter], columns=['x','y'])
    df['label'] = labels
    scat_plot = df.hvplot.scatter(x='x',y='y',color='label',aspect='square', cmap=cmap, fontsize=16, frame_width=400).opts(legend_position='top')
    cost_curve = pd.Series(cost).hvplot(title='Objective Function', xlabel='Gradient Descent Iteration', ylabel='KL Divergence', fontsize=16) * \
                 hv.VLine(early_exaggeration_ends).opts(color='r', line_width=.5) * \
                 hv.VLine(gd_iter).opts(color='k',line_dash='dashed', line_width=0.5)
    return scat_plot + cost_curve

dmap = hv.DynamicMap(return_map_for_iter, kdims=['GD_iteration'])
dmap.redim.range(GD_iteration=(0,n_iter))
# -


hv.Points((8,gd_obj_df[8])).opts(size=50, color='r')



Q_iter5 = plot_matrix(Q_dict[5], q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$');

Q_iter30 = plot_matrix(Q_dict[30], q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$')

Q_iter50 = plot_matrix(Q_dict[50], q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$')

Q_iter100 = plot_matrix(Q_dict[100], q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$')

Q_iter500 = plot_matrix(Q_dict[500], q_min=0.05, q_max=0.90, ctitle='$Q_{j|i}$')

Y_df = pd.DataFrame(Y,columns=['x','y'])
if isinstance(labels[0],str):
    Y_df['label'] = labels 
else:
    Y_df['label'] = [str(int(i)) for i in labels]
Y_df['time'] = np.arange(len(labels))
Y_df.infer_objects();

Y_df.hvplot.scatter(x='x',y='y', color='label', aspect='square', cmap=cmap) + Y_df.hvplot.scatter(x='x',y='y', color='time', aspect='square', cmap='RdBu')





















# ***
# # SKLEARN TSNE FOR COMPARISON

tsne_obj = TSNE(n_components=2, perplexity=desired_perplexity, early_exaggeration = early_exaggeration, 
                learning_rate=learning_rate, n_iter=n_iter, n_iter_without_progress=n_iter, 
                min_grad_norm=1e-25,
                metric='euclidean',
                init='random', random_state=43, method='exact', n_jobs=-1, verbose = 2)

# %%time
tsne_sklearn = tsne_obj.fit_transform(X)

tsne_sklearn = pd.DataFrame(tsne_sklearn,columns=['x','y'])
if isinstance(labels[0],str):
    tsne_sklearn['label'] = labels 
else:
    tsne_sklearn['label'] = [str(int(i)) for i in labels]
tsne_sklearn['time'] = np.arange(len(labels))
tsne_sklearn.infer_objects();

tsne_sklearn.hvplot.scatter(x='x',y='y', color='label', aspect='square', cmap=cmap) + tsne_sklearn.hvplot.scatter(x='x',y='y', color='time', aspect='square', cmap='RdBu')

SKLEARN_P = np.loadtxt('/data/SFIMJGC_HCP7T/manifold_learning_fmri/Resources/Figure04/SKLEARN_P.txt')
SKLEARN_D = pdist(X, 'euclidean')

df = pd.DataFrame(columns=['D','P'])

df['D'] = SKLEARN_D
df['P'] = SKLEARN_P

sns.displot(data=df,x='D')

sns.displot(data=df,x='P')

squareform(SKLEARN_P).sum()

sns.jointplot(data=df,x='D',y='P', hue='labels', cmap='jet')

# + tags=[]
if CORRECT_FC == True:
    WL_samples = 30
    WS_samples = 1
    Nwins = X.shape[0]
    t = np.arange(Nwins)
    a=.01
    b=-.3
    c=1.1
    win_overlaps = (c / (1+a*np.exp(-b*t)) )+1
    corr_factor = np.zeros((Nwins,Nwins))
    for i in np.arange(Nwins):
        corr_factor[i,:] = shift(win_overlaps,i) + corr_factor[i,:]
    corr_factor = corr_factor + corr_factor.T
    np.fill_diagonal(corr_factor,corr_factor[0,0]/2)
    pd.DataFrame(corr_factor).loc[0,:].hvplot(xlim=[0,200]) * hv.VLine(30)
    D = np.multiply(D,corr_factor)

# + tags=[]
if CORRECT_FC == True:
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    mat = sns.heatmap(corr_factor, vmin=np.quantile(corr_factor,0.01), vmax=np.quantile(corr_factor,0.99), square=True, cmap='viridis')
    if DATASET == 'FC':
        for idx in line_idxs:
            ax.plot([idx,idx],[0,corr_factor.shape[0]],'w--')
            ax.plot([0,corr_factor.shape[0]],[idx,idx],'w--')
        ax.yaxis.set_ticks(tick_idxs);
        ax.xaxis.set_ticks(tick_idxs);
        ax.yaxis.set_ticklabels(tick_labels,fontsize=18);
        ax.xaxis.set_ticklabels(tick_labels,fontsize=18);
        cbar = mat.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.label.set_size(26)
        cbar.ax.yaxis.set_label_position('left')

# + tags=[]
if CORRECT_FC == True:
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    mat = sns.heatmap(D, vmin=np.quantile(D,0.05), vmax=np.quantile(D,0.95), square=True, cmap='viridis')
    if DATASET == 'FC':
        for idx in line_idxs:
            ax.plot([idx,idx],[0,D.shape[0]],'w--')
            ax.plot([0,D.shape[0]],[idx,idx],'w--')
        ax.yaxis.set_ticks(tick_idxs);
        ax.xaxis.set_ticks(tick_idxs);
        ax.yaxis.set_ticklabels(tick_labels,fontsize=18);
        ax.xaxis.set_ticklabels(tick_labels,fontsize=18);
        cbar = mat.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.label.set_size(26)
        cbar.ax.yaxis.set_label_position('left')
# -

# #### 1.2. Estimate Variance per point that leads to the selected perplexity

# + tags=[]
var_dict, perp_search_dict = gauss_var_search(D,desired_perplexity,tol=1e-5,n_attempts=50, beta_init=1)
# -

# Show distribution of actual perplexity to make sure there are no issues
aux_p = [perp_search_dict[i][-1] for i in range(D.shape[0])]
aux_p = pd.Series(aux_p, name='Perplexity')
aux_v = [var_dict[k] for k in var_dict.keys()]
aux_v = pd.Series(aux_v, name='Variance')
aux_p.hvplot.kde(title='Distribution of Perplexity estimates', xlabel='Perplexity Estimates', ylabel='Density', fontsize=16) * aux_p.hvplot.hist(normed=True) + \
aux_v.hvplot.kde(title='Distribution of Variance estimates', ylabel='Density', xlabel='Variance', fontsize=16) * aux_v.hvplot.hist(normed=True)

# #### 1.3 Estimate the Similarity Matrix using point-specific variances just estimated

P = get_P(D,var_dict)

i = 270
n = D.shape[0]
selection = list(np.concatenate((np.r_[0:i], np.r_[i+1:n])))
Di  = D[i, selection]
Pi  = P[i, selection]
aux = pd.DataFrame([Di,Pi], index=['D','P']).T
aux['label'] = labels[selection].values
#.hvplot.scatter(x='D',y='P')

aux.hvplot.scatter(x='D',y='P', color='label', alpha=0.5, cmap=cmap)

fig,ax = plt.subplots(1,1,figsize=(10,8))
mat = sns.heatmap(P, vmin=np.quantile(P,0.05), vmax=np.quantile(P,0.95), square=True, cmap='viridis')
if DATASET == 'FC' or DATASET == 'MNIST_SORTED':
    for idx in line_idxs:
        ax.plot([idx,idx],[0,P.shape[0]],'w--')
        ax.plot([0,P.shape[0]],[idx,idx],'w--')
    ax.yaxis.set_ticks(tick_idxs);
    ax.xaxis.set_ticks(tick_idxs);
    ax.yaxis.set_ticklabels(tick_labels,fontsize=18);
    ax.xaxis.set_ticklabels(tick_labels,fontsize=18);
    cbar = mat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(26)
    cbar.ax.yaxis.set_label_position('left')

# Now we make sure that the matrix is symetric

P = P + np.transpose(P)
P = P / np.sum(P)

fig,ax = plt.subplots(1,1,figsize=(10,8))
mat = sns.heatmap(P, vmin=np.quantile(P,0.05), vmax=np.quantile(P,0.95), square=True, cmap='viridis')
if DATASET == 'FC' or DATASET == 'MNIST_SORTED':
    for idx in line_idxs:
        ax.plot([idx,idx],[0,P.shape[0]],'w--')
        ax.plot([0,P.shape[0]],[idx,idx],'w--')
    ax.yaxis.set_ticks(tick_idxs);
    ax.xaxis.set_ticks(tick_idxs);
    ax.yaxis.set_ticklabels(tick_labels,fontsize=18);
    ax.xaxis.set_ticklabels(tick_labels,fontsize=18);
    cbar = mat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(26)
    cbar.ax.yaxis.set_label_position('left')

# An early exxageration step is also implemented to help with the upcoming optimization problem.

P = P * early_exaggeration # early exaggeration
P = np.maximum(P, 1e-12)
np.fill_diagonal(P,0)

fig,ax = plt.subplots(1,1,figsize=(10,8))
mat = sns.heatmap(P, vmin=np.quantile(P,0.05), vmax=np.quantile(P,0.95), square=True, cmap='viridis')
if DATASET == 'FC' or DATASET == 'MNIST_SORTED':
    for idx in line_idxs:
        ax.plot([idx,idx],[0,P.shape[0]],'w--')
        ax.plot([0,P.shape[0]],[idx,idx],'w--')
    ax.yaxis.set_ticks(tick_idxs);
    ax.xaxis.set_ticks(tick_idxs);
    ax.yaxis.set_ticklabels(tick_labels,fontsize=18);
    ax.xaxis.set_ticklabels(tick_labels,fontsize=18);
    cbar = mat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.label.set_size(26)
    cbar.ax.yaxis.set_label_position('left')

# + tags=[]
pd.DataFrame(squareform(P)).hvplot.hist(normed=True, bins=500) * pd.DataFrame(squareform(P)).hvplot.kde(title='Distribution of similarities in original space')

# + tags=[]
# Run iterations
Y_dict={}
Y_dict[0] = Y
cost = []
for iter in tqdm(range(n_iter)):
    # Using Eq.4 from the Van Maaten paper
    D_low = squareform(pdist(Y, 'euclidean'))
    Q_num = 1/ (1 + np.square(D_low))
    np.fill_diagonal(Q_num,0.)
    Q_den = np.sum(Q_num)
    Q     = Q_num / Q_den
    Q = np.maximum(Q, 1e-12) # To avoid divisions by zero
    #print(Q_num[0,:])
    #print(Q_den)
    #print(Q[0,:])
    #if iter==1:
    #    fsf
    # Compute gradient (Eq. 5)
    PQ = P - Q
    for i in range(n):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * Q_num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)
    
    # Perform the update
    if iter < 20:
        momentum = initial_momentum
    else:
        momentum = final_momentum
    gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
            (gains * 0.8) * ((dY > 0.) == (iY > 0.))
    gains[gains < min_gain] = min_gain
    iY = momentum * iY - learning_rate * (gains * dY)
    Y = Y + iY
    Y = Y - np.tile(np.mean(Y, 0), (n, 1))
    Y_dict[iter+1] = Y
    
    # Compute current value of cost function
    C = np.sum(P * np.log(P / Q))
    #if (iter + 1) % 50 == 0:
    #    print("Iteration %d: error is %f" % (iter + 1, C))
    cost.append(C)
    
    # End of early exaggeration (in the paper it says to stop after 50 iterations, here it is after 100)
    if iter == early_exaggeration_ends:
        P = P / early_exaggeration
# -















from streamz.dataframe import DataFrames

# +
# DataFrames?
# -

Q = num / np.sum(num)
Q = np.maximum(Q, 1e-12)
print(Q.shape)
Q

D_low = squareform(pdist(Y, 'euclidean'))
Q_num = 1/ (1 + np.square(D_low))
Q_den = np.sum(Q_num)
Q     = Q_num / Q_den
Q

Q_den = np.sum(Q_num[np.triu_indices(n,1)])
Q     = Q_num / Q_den
Q

np.sum(Q_num[np.triu_indices(n,1)])

Y



# Although original formulations are written in terms of variance and the actual perplexity, to minimize numerical errors and avoid divisions by zero, the code below works in terms of two other quantitites:
#
# * blearning_rate = (1 / 2) * variance
# * H = np.log(perplexity)

gauss_var_search(D,perplexity,tol=1e-5,n_attempts=50, blearning_rate_init=1):

# +
perplexities = {}
# Selection of Gamma for each data point based on the Perplexity and a Gaussian centered on that point
i=0

# Grab all distances from point i to all other points.
Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] # Entries in a row except the diagonal element (i,i) that is zero --> Euclidean distance between i and all other datapoints

# Initial values for blearning_rate
blearning_ratemin = -np.inf
blearning_ratemax = np.inf
# Compute initial perplexity for initial variance = 0.5
perplexity_init = get_perplexity(Di, var_init)
perplexities[i] = [perplexity_init]

# Compute difference between perplexity_init and desired perplexity [we do this in terms of the natural logs]
Hdiff = np.log(perplexity_init) - np.log(perplexity)
tries = 0

# Keep updating the variance, until we get to a perplexity that is within tolerance limits to the desired one or we hit the maximum number of attempts
while np.abs(Hdiff) > perp_tol and tries < perp_n_attempts:
    if Hdiff > 0:
        blearning_ratemin = blearning_rate[i].copy()
        if blearning_ratemax == np.inf or blearning_ratemax == -np.inf:
            blearning_rate[i] = blearning_rate[i] * 2.
        else:
            blearning_rate[i] = (blearning_rate[i] + blearning_ratemax) / 2.
    else:
        blearning_ratemax = blearning_rate[i].copy()
        if blearning_ratemin == np.inf or blearning_ratemin == -np.inf:
            blearning_rate[i] = blearning_rate[i] / 2.
        else:
            blearning_rate[i] = (blearning_rate[i] + blearning_ratemin) / 2.
    
    # Recompute the values
    new_var    = 1 / (2*blearning_rate[i]) 
    this_perplexity = get_perplexity(Di,new_var)
    this_P          = get_Pij_row(Di, new_var)
    Hdiff           = np.log(this_perplexity) - np.log(perplexity)
    tries          += 1
    perplexities[i].append(this_perplexity)
# -

perplexities

# +
# Compute one row of P and its perplexity for blearning_rate = 1 | blearning_rate = 1/2 * precision = 1/2 * 1/variance = 1/2 * 1 / sigma**2
blearning_ratemin = -np.inf
blearning_ratemax = np.inf
(H, thisP) = Hblearning_rate(Di, blearning_rate[i])
perplexities.append(np.exp(H[0]))

# Evaluate whether the perplexity is within tolerance
Hdiff = H - logU
tries = 0
while np.abs(Hdiff) > tol and tries < 50:
    # If not, increase or decrease precision
    if Hdiff > 0:
        blearning_ratemin = blearning_rate[i].copy()
        if blearning_ratemax == np.inf or blearning_ratemax == -np.inf:
            blearning_rate[i] = blearning_rate[i] * 2.
        else:
            blearning_rate[i] = (blearning_rate[i] + blearning_ratemax) / 2.
    else:
        blearning_ratemax = blearning_rate[i].copy()
        if blearning_ratemin == np.inf or blearning_ratemin == -np.inf:
            blearning_rate[i] = blearning_rate[i] / 2.
        else:
            blearning_rate[i] = (blearning_rate[i] + blearning_ratemin) / 2.

    # Recompute the values
    (H, thisP) = Hblearning_rate(Di, blearning_rate[i])
    perplexities.append(np.exp(H[0]))
    Hdiff = H - logU
    tries += 1
# -





# One input is the targetted perplexity, which will be used to compute the actual variance of the distribution on each point. By fixing the perplexity, instead of the variance, we become robust against variations of denstity across points

logU = np.log(perplexity) # Instead of working directly with the perplexity, we will work with its natural logarithm --> logU is the target value

# We now enter a loop in which we compute perplexity and conditional probability for point i

# +
perplexities = []
# Selection of Gamma for each data point based on the Perplexity and a Gaussian centered on that point
i=0
# Grab all distances from point i to all other points.
Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] # Entries in a row except the diagonal element (i,i) that is zero --> Euclidean distance between i and all other datapoints

# Compute one row of P and its perplexity for blearning_rate = 1 | blearning_rate = 1/2 * precision = 1/2 * 1/variance = 1/2 * 1 / sigma**2
blearning_ratemin = -np.inf
blearning_ratemax = np.inf
(H, thisP) = Hblearning_rate(Di, blearning_rate[i])
perplexities.append(np.exp(H[0]))

# Evaluate whether the perplexity is within tolerance
Hdiff = H - logU
tries = 0
while np.abs(Hdiff) > tol and tries < 50:
    # If not, increase or decrease precision
    if Hdiff > 0:
        blearning_ratemin = blearning_rate[i].copy()
        if blearning_ratemax == np.inf or blearning_ratemax == -np.inf:
            blearning_rate[i] = blearning_rate[i] * 2.
        else:
            blearning_rate[i] = (blearning_rate[i] + blearning_ratemax) / 2.
    else:
        blearning_ratemax = blearning_rate[i].copy()
        if blearning_ratemin == np.inf or blearning_ratemin == -np.inf:
            blearning_rate[i] = blearning_rate[i] / 2.
        else:
            blearning_rate[i] = (blearning_rate[i] + blearning_ratemin) / 2.

    # Recompute the values
    (H, thisP) = Hblearning_rate(Di, blearning_rate[i])
    perplexities.append(np.exp(H[0]))
    Hdiff = H - logU
    tries += 1
# -

perplexities

pd.DataFrame(X[2,:].reshape(10,5)).hvplot.heatmap(cmap='gray', aspect='square')

# +
i=2
# Grab all distances from point i to all other points.
blearning_ratemin = -np.inf #
blearning_ratemax = np.inf
Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] # Entries in a row except the diagonal element (i,i) that is zero --> Euclidean distance between i and all other datapoints

# Start with a precision of 1
(H, thisP) = Hblearning_rate(Di, blearning_rate[i])
# -

H

# Compute the sigma_i for this point given the perplexity
P    = np.exp(-Di.copy() * blearning_rate[i])
sumP = sum(P)                          # Denominator of Pi,j
H = np.log(sumP) + blearning_rate * np.sum(D * P) / sumP
P = P / sumP

P    = np.exp(-Di.copy() * blearning_rate[i])

P.shape

D[i,:]

Di


def Hblearning_rate(D=np.array([]), blearning_rate=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * blearning_rate)
    sumP = sum(P)
    H = np.log(sumP) + blearning_rate * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    blearning_rate = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        blearning_ratemin = -np.inf
        blearning_ratemax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hblearning_rate(Di, blearning_rate[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                blearning_ratemin = blearning_rate[i].copy()
                if blearning_ratemax == np.inf or blearning_ratemax == -np.inf:
                    blearning_rate[i] = blearning_rate[i] * 2.
                else:
                    blearning_rate[i] = (blearning_rate[i] + blearning_ratemax) / 2.
            else:
                blearning_ratemax = blearning_rate[i].copy()
                if blearning_ratemin == np.inf or blearning_ratemin == -np.inf:
                    blearning_rate[i] = blearning_rate[i] / 2.
                else:
                    blearning_rate[i] = (blearning_rate[i] + blearning_ratemin) / 2.

            # Recompute the values
            (H, thisP) = Hblearning_rate(Di, blearning_rate[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / blearning_rate)))
    return P










sum_X = np.sum(np.square(X), 1)
D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

sum_X.shape

D.shape

D

DS = squareform(pdist(X, 'euclidean'))
#DS = pd.DataFrame(DS,columns=win_labels, index=win_labels)

DS*DS


def Hblearning_rate(D=np.array([]), blearning_rate=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * blearning_rate)
    sumP = sum(P)
    H = np.log(sumP) + blearning_rate * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    blearning_rate = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        blearning_ratemin = -np.inf
        blearning_ratemax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hblearning_rate(Di, blearning_rate[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                blearning_ratemin = blearning_rate[i].copy()
                if blearning_ratemax == np.inf or blearning_ratemax == -np.inf:
                    blearning_rate[i] = blearning_rate[i] * 2.
                else:
                    blearning_rate[i] = (blearning_rate[i] + blearning_ratemax) / 2.
            else:
                blearning_ratemax = blearning_rate[i].copy()
                if blearning_ratemin == np.inf or blearning_ratemin == -np.inf:
                    blearning_rate[i] = blearning_rate[i] / 2.
                else:
                    blearning_rate[i] = (blearning_rate[i] + blearning_ratemin) / 2.

            # Recompute the values
            (H, thisP) = Hblearning_rate(Di, blearning_rate[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / blearning_rate)))
    return P


print("Run Y = tsne.tsne(X, n_components, perplexity) to perform t-SNE on your dataset.")
print("Running example on 2,500 MNIST digits...")
X      = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")

# Run PCA to initialize the mapping
X = PCA(n_components=50,svd_solver='full').fit_transform(X)

# Other initializations
n_components = 2
perplexity   = 30.0
(n, d) = X.shape
n_iter = 1000
initial_momentum = 0.5
final_momentum = 0.8
learning_rate = 500
min_gain = 0.01
Y = np.random.randn(n, n_components)
dY = np.zeros((n, n_components))
iY = np.zeros((n, n_components))
gains = np.ones((n, n_components))

# Compute P-values
P = x2p(X, 1e-5, perplexity)
P = P + np.transpose(P)
P = P / np.sum(P)
P = P * 4. # early exaggeration
P = np.maximum(P, 1e-12)

sns.heatmap(P, vmin=1e-12, vmax=1e-10, cmap='viridis')

pd.DataFrame(P).melt().drop('variable',axis=1).plot.kde()

# + tags=[]
# Run iterations
for iter in range(n_iter):

    # Compute pairwise affinities
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)

    # Compute gradient
    PQ = P - Q
    for i in range(n):
        dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)

    # Perform the update
    if iter < 20:
        momentum = initial_momentum
    else:
        momentum = final_momentum
    gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
    gains[gains < min_gain] = min_gain
    iY = momentum * iY - learning_rate * (gains * dY)
    Y = Y + iY
    Y = Y - np.tile(np.mean(Y, 0), (n, 1))

    # Compute current value of cost function
    if (iter + 1) % 10 == 0:
        C = np.sum(P * np.log(P / Q))
        print("Iteration %d: error is %f" % (iter + 1, C))

    # Stop lying about P-values
    if iter == 100:
        P = P / 4.
# -

pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()

Y_df = pd.DataFrame(Y, columns=['x','y'])
Y_df.infer_objects()
Y_df['label'] = [str(int(l)) for l in labels]
Y_df

Y_df.hvplot.scatter(x='x',y='y', color='label', aspect='square')

from sklearn.manifold import TSNE

Y_sklearn = TSNE(init='pca',verbose=10, n_components=50, method='exact').fit_transform(X)

Y_sklearn = pd.DataFrame(Y_sklearn, columns=['x','y'])
Y_sklearn.infer_objects()
Y_sklearn['label'] = [str(int(l)) for l in labels]
Y_sklearn

Y_sklearn.hvplot.scatter(x='x',y='y', color='label', aspect='square')


def tsne(X=np.array([]), n_components=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to n_components dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, n_components, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(n_components, float):
        print("Error: array X should have type float.")
        return -1
    if round(n_components) != n_components:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    n_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    learning_rate = 500
    min_gain = 0.01
    Y = np.random.randn(n, n_components)
    dY = np.zeros((n, n_components))
    iY = np.zeros((n, n_components))
    gains = np.ones((n, n_components))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(n_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - learning_rate * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y



Y = tsne(X, 2, 50, 20.0)
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
pylab.show()
