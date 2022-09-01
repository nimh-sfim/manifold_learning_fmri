import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import hvplot.networkx as hvnx

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)
 
def plot_matrix(m,tick_labels=None, tick_idxs=None, line_idxs=None, q_min=0.01, q_max=0.99, cmap='viridis', ctitle='Euclidean Distance', figsize=(11,8), lab_fsize=18, net_separators='w--', cticks=None, clabels=None):
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
          ax.plot([idx,idx],[0,m.shape[0]],net_separators);
          ax.plot([0,m.shape[0]],[idx,idx],net_separators);
    if (not(tick_idxs is None)) & (not(tick_labels is None)): 
       ax.yaxis.set_ticks(tick_idxs);
       ax.xaxis.set_ticks(tick_idxs);
       ax.yaxis.set_ticklabels(tick_labels,fontsize=lab_fsize);
       ax.xaxis.set_ticklabels(tick_labels,fontsize=lab_fsize);
    cbar = mat.collections[0].colorbar;
    
    if cticks is not None:
        cbar.ax.yaxis.set_ticks(cticks)
    if clabels is not None:
        cbar.ax.yaxis.set_ticklabels(clabels)
    
    cbar.ax.tick_params(labelsize=16);
    cbar.ax.yaxis.label.set_size(20);
    cbar.ax.yaxis.set_label_position('left');
    plt.close()
    return fig
   
def plot_matrix_as_graph(M, layout='spectral', edge_color='gray', edge_width=0.1, node_size=50, arrowhead_length=0.01, node_labels=None, node_cmap=None, node_color='white', verbose=False, no_edges=False, random_state=43, height=500, width=500, toolbar=None):
    # Check if provided matrix is symmetric
    # =====================================
    M_is_sym = check_symmetric(M)
    if verbose:
        print("++ INFO[plot_matrix_as_graph]: Input Matrix Symmetric? %s" % str(M_is_sym))
    # Create Graph Object
    # ===================
    if M_is_sym:
        G = nx.from_numpy_matrix(M)
    else:
        G = nx.from_numpy_matrix(M, create_using=nx.DiGraph)
    if verbose:
        print('++ INFO[plot_matrix_as_graph]: Graph Object created [type=%s]' % str(type(G)))
    # Create Layout
    # =============
    if layout == 'spectral':
        G_pos = nx.layout.spectral_layout(G)
    elif layout == 'spring':
        G_pos = nx.layout.spring_layout(G, seed=random_state)
    else:
        G_pos = nx.layout.circular_layout(G)
    if verbose:
        print('++ INFO[plot_matrix_as_graph]: Graph positions computed')
    # Plot the Graph Edges
    # ====================
    if no_edges is True:
        G_plot = None
    else:
        if verbose:
           print('++ INFO[plot_matrix_as_graph]: Plotting the edges....', end='')
        if M_is_sym:
           G_plot = hvnx.draw(G,G_pos,edge_width=edge_width, edge_color=edge_color, node_size=node_size, arrowhead_length=arrowhead_length, width=width, height=height).opts(toolbar=toolbar)
        else:
           G_plot = hvnx.draw(G,G_pos,edge_width=edge_width, edge_color=edge_color, node_size=node_size, width=width, height=height).opts(toolbar=toolbar)
        if verbose:
           print(' [DONE]')
    # Plot the Graph Nodes
    # ====================
    if verbose:
        print('++ INFO[plot_matrix_as_graph]: Plotting the nodes....', end='')
    if node_labels is not None:
        unique_labels = node_labels.unique()
        unique_labels.sort()
        for label in unique_labels:
            if G_plot is None:
                G_plot = hvnx.draw_networkx_nodes(G, G_pos, nodelist=np.where(node_labels == label)[0].tolist(), node_color=node_cmap[label], node_size=node_size, width=width, height=height).opts(toolbar=toolbar)
            else:
                G_plot = G_plot * hvnx.draw_networkx_nodes(G, G_pos, nodelist=np.where(node_labels == label)[0].tolist(), node_color=node_cmap[label], node_size=node_size)
    else:
        if G_plot is None:
           G_plot = hvnx.draw_networkx_nodes(G, G_pos, node_color=node_color, node_size=node_size)
        else:
           G_plot = G_plot * hvnx.draw_networkx_nodes(G, G_pos, node_color=node_color, node_size=node_size)
    if verbose:
        print(' [DONE]')
    # Return Final Object
    return G_plot