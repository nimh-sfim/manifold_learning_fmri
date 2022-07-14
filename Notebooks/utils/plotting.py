import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matrix(m,tick_labels=None, tick_idxs=None, line_idxs=None, q_min=0.01, q_max=0.99, cmap='viridis', ctitle='Euclidean Distance', figsize=(11,8), lab_fsize=18):
    """This function will plot a dissimilarity or affinity matrix and annotate it based on available labels
    
    Inputs:
    -------
    m: matrix to plot
    q_min: quantile used to set the minimum value on the colorscale (0 - 1)
    q_max: quantile used to set the maximum value on the colorscale (0 - 1)
    cmap: colormap
    cttile: string to accompany the colorbar
    figsize: figure size
    lab_fsize: font size for labels
    line_idxs: location of dashed line separators
    tick_idxs: tick indexes
    tick_labels: tick labels
    
    Returns:
    --------
    A plot of the matrix
    """
    fig,ax = plt.subplots(1,1,figsize=figsize);
    mat = sns.heatmap(m, vmin=np.quantile(m,q_min), vmax=np.quantile(m,q_max), square=True, cmap=cmap,cbar_kws={'label': ctitle});
    if not(line_idxs is None):
       for idx in line_idxs:
          ax.plot([idx,idx],[0,m.shape[0]],'w--');
          ax.plot([0,m.shape[0]],[idx,idx],'w--');
    if (not(tick_idxs is None)) & (not(tick_labels is None)): 
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