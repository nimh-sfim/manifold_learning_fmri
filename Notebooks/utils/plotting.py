import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import hvplot.networkx as hvnx
import plotly.express as px
import hvplot.pandas
import panel as pn
import os
import os.path as osp
from .basics import task_cmap_caps as task_cmap
from .basics import PRJ_DIR, group_method_2_label
from .basics import norm_methods
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
HUGE_SIZE   = 22

NULL_CONNRAND_PALETTE  = sns.color_palette('Wistia',n_colors=3)
NULL_PHASERAND_PALETTE = sns.color_palette('gist_gray',n_colors=3)
ORIGINAL_PALETTE       = sns.color_palette(palette='bright',n_colors=3)

scene_default_3d = dict(
        xaxis = dict(nticks=4, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=4, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=4, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))

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

def plot_2d_scatter(data,x,y,c,cmap=task_cmap, show_frame=False, s=2, alpha=0.3, toolbar=None, legend=False, xaxis=False, xlabel='', yaxis=False, ylabel='', frame_width=250, shared_axes=False):
    plot = data.hvplot.scatter(x=x,y=y,c=c, cmap=cmap, 
                            aspect='square', s=s, alpha=alpha, 
                            legend=legend, xaxis=xaxis, 
                            yaxis=yaxis, frame_width=frame_width, shared_axes=shared_axes).opts(toolbar=toolbar, show_frame=show_frame, tools=[])
    return plot
camera = dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=1)) 

def plot_3d_scatter(data,x,y,z,c,cmap,s=2,width=250, height=250, ax_range=[-.005,.005],nticks=4):
    fig = px.scatter_3d(data,
                        x=x,y=y,z=z, 
                        width=width, height=height, 
                        opacity=0.3, color=c,color_discrete_sequence=cmap)
    fig.update_layout(showlegend=False, 
                          font_color='white');
    scene_extra_confs = dict(
        xaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        yaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),
        zaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))
    fig.update_layout(scene_camera=camera, scene=scene_extra_confs, scene_aspectmode='cube',margin=dict(l=2, r=2, b=0, t=0, pad=0))
    fig.update_traces(marker_size = s)
    return fig

def get_SIvsKNN_plots(data,x,y,hue=None,palette=None,style=None,style_order=None, 
                      xticks=[30,65,100,135,170,200],font_scale=1.5,figsize=(10,5), 
                      xlabel='',ylabel='',title='',grid_vis=True, hue_order=None, ylim=(-.2,0.9)):
    fig,ax = plt.subplots(1,1,figsize=figsize)
    g      = sns.lineplot(data=data,x=x,y=y, hue=hue, palette=palette,  style=style, style_order=style_order, ax=ax)
    g.set_xlabel(xlabel)
    g.set_ylabel(ylabel)
    g.legend(loc='lower right', ncol=2, fontsize=10)
    g.set_xticks(xticks)
    g.set_title(title)
    g.set(ylim=ylim)
    g.yaxis.tick_right()
    g.grid(grid_vis)
    plt.close()
    return fig

def generate_Avg_LE_SIvsKnn_ScanLevel(si_LE,sbj_list, figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for m in m_list:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        data    = si_LE.loc[sbj_list].loc[:,'Original',:,:,:,m,:].reset_index()
        data_nc = si_LE.loc[sbj_list].loc[:,'Null_ConnRand',:,:,:,m,:].reset_index()
        data_np = si_LE.loc[sbj_list].loc[:,'Null_PhaseRand',:,:,:,m,:].reset_index()
        g_orig = sns.lineplot(data=data,y='SI',x='Knn', hue='Metric', style='Norm', ax=ax,  palette=ORIGINAL_PALETTE)
        g_nc = sns.lineplot(data=data_nc,y='SI',x='Knn', hue='Metric', style='Norm', ax=ax, palette=NULL_CONNRAND_PALETTE ,  legend=False)
        g_np = sns.lineplot(data=data_np,y='SI',x='Knn', hue='Metric', style='Norm', ax=ax, palette=NULL_PHASERAND_PALETTE, legend=False)
        ax.set_ylim(y_min,y_max)
        ax.set_xlim(x_min,x_max)
        ax.grid()
        ax.set_title('Scan-Level | {m}D'.format(m=str(m)), fontsize=HUGE_SIZE)
        ax.legend(loc='upper right', ncol=2)
        ax.set_ylabel('$SI_{%s}$' % 'Task')
        #ax.set_ylabel('SI [{target}]'.format(target='Task'))
        out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','LE')
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
            if verbose:
               print('++ INFO: Folder created [%s]' % out_dir)
        out_path = osp.join(out_dir,'SIvsKNN_ScanLevel_AVG_m{m}_{target}.png'.format(m=str(m), target='Task'))
        plt.savefig(out_path,bbox_inches='tight')
        if verbose:
            print('++ INFO: Figure saved to disk [%s]' % out_path)
        plt.close()
    return None

def generate_LE_SIvsKnn_ScanLevel(si_LE,sbj,figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for m in m_list:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        data    = si_LE.loc[sbj,'Original',:,:,:,m,:].reset_index()
        data_nc = si_LE.loc[sbj,'Null_ConnRand',:,:,:,m,:].reset_index()
        data_np = si_LE.loc[sbj,'Null_PhaseRand',:,:,:,m,:].reset_index()
        g_orig = sns.lineplot(data=data,y='SI',x='Knn', hue='Metric', style='Norm', ax=ax,  palette=ORIGINAL_PALETTE)
        g_nc = sns.lineplot(data=data_nc,y='SI',x='Knn', hue='Metric', style='Norm', ax=ax, palette=NULL_CONNRAND_PALETTE ,  legend=False)
        g_np = sns.lineplot(data=data_np,y='SI',x='Knn', hue='Metric', style='Norm', ax=ax, palette=NULL_PHASERAND_PALETTE, legend=False)
        ax.set_ylim(y_min,y_max)
        ax.set_xlim(x_min,x_max)
        ax.grid()
        ax.set_title('Scan-Level [{sbj}] | {m}D'.format(m=str(m),sbj=sbj), fontsize=HUGE_SIZE)
        ax.legend(loc='upper right', ncol=2)
        ax.set_ylabel('$SI_{%s}$' % 'Task')
        out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','LE')
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
            if verbose:
               print('++ INFO: Folder created [%s]' % out_dir)
        out_path = osp.join(out_dir,'SIvsKNN_ScanLevel_{sbj}_m{m}_{target}.png'.format(m=str(m), target='Task',sbj=sbj))
        plt.savefig(out_path,bbox_inches='tight')
        if verbose:
            print('++ INFO: Figure saved to disk [%s]' % out_path)
        plt.close()
    return None

def generate_LE_SIvsKNN_GroupLevel(si_LE,comb_methods=['ALL','Procrustes'],figsize=(10,5),target_m_tuples=[('Task',2),('Task',3),('Subject',2),('Subject',3)], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for group_method in comb_methods:
        for (target,m) in target_m_tuples:
            fig, ax = plt.subplots(1,1,figsize=(10,5))
            data_orig = si_LE.loc[group_method].loc['Original',:,:,:,m,target].reset_index()
            data_nc   = si_LE.loc[group_method].loc['Null_ConnRand',:,:,:,m,target].reset_index()
            data_np   = si_LE.loc[group_method].loc['Null_PhaseRand',:,:,:,m,target].reset_index()
            if target == 'Task':
                pal_orig = ORIGINAL_PALETTE
                pal_nc   = NULL_CONNRAND_PALETTE 
                pal_np   = NULL_PHASERAND_PALETTE
            else:
                pal_orig = ORIGINAL_PALETTE
                pal_nc   = NULL_CONNRAND_PALETTE 
                pal_np   = NULL_PHASERAND_PALETTE

            sns.lineplot(data=data_orig,y='SI',x='Knn', hue='Metric', style='Norm',  ax=ax, palette=pal_orig)
            sns.lineplot(data=data_nc,  y='SI',x='Knn', hue='Metric', style='Norm',  ax=ax, palette=pal_nc, legend=False)
            sns.lineplot(data=data_np,  y='SI',x='Knn', hue='Metric', style='Norm',  ax=ax, palette=pal_np, legend=False)
            ax.set_ylim(y_min,y_max)
            ax.set_xlim(x_min,x_max)
            ax.grid()
            ax.set_title('Group-Level [{gm}] | {m}D'.format(m=str(m),gm=group_method_2_label[group_method]), fontsize=HUGE_SIZE)
            ax.legend(loc='upper right', ncol=2)
            ax.set_ylabel('SI [{target}]'.format(target=target))
            out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','LE')
            if not osp.exists(out_dir):
                os.makedirs(out_dir)
                if verbose:
                    print('++ INFO: Folder created [%s]' % out_dir)
            out_path = osp.join(out_dir,'SIvsKNN_GroupLevel_{gm}_m{m}_{target}.png'.format(m=str(m), target=target, gm=group_method))
            plt.savefig(out_path,bbox_inches='tight')
            if verbose:
                print('++ INFO: Figure saved to disk [%s]' % out_path)
            plt.close()
    return None
   
#def generate_LE_SIvsKNN_GroupLevel_Rdist(si_LE,comb_methods=['ALL','Procrustes'],figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200):
#    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
#    for group_method in comb_methods:
#        for m in m_list:
#            fig, ax   = plt.subplots(1,1,figsize=figsize)
#            data_orig = si_LE.loc[group_method].loc['Original',:,'correlation',:,m,:].reset_index()
#            data_nc   = si_LE.loc[group_method].loc['Null_ConnRand',:,'correlation',:,m,:].reset_index()
#            data_np   = si_LE.loc[group_method].loc['Null_PhaseRand',:,'correlation',:,m,:].reset_index()
#            sns.lineplot(data=data_orig, y='SI',x='Knn',hue='Target',style='Norm', palette=sns.color_palette('hls',n_colors=2), ax=ax, linewidth=5)
#            sns.lineplot(data=data_nc, y='SI',x='Knn',hue='Target',style='Norm', palette=NULL_CONNRAND_PALETTE , ax=ax, legend=False)
#            sns.lineplot(data=data_np, y='SI',x='Knn',hue='Target',style='Norm', palette=NULL_PHASERAND_PALETTE, ax=ax, legend=False)
#            ax.set_ylim(y_min,y_max)
#            ax.set_xlim(x_min,x_max)
#            ax.grid()
#            ax.set_title('Group-Level [{gm}] | {m}D | Correlation Distance'.format(m=str(m),gm=group_method_2_label[group_method]), fontsize=HUGE_SIZE)
#            ax.legend(loc='upper right', ncol=2)
#            ax.set_ylabel('SI')
#            out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','LE')
#            if not osp.exists(out_dir):
#                os.makedirs(out_dir)
#                if verbose:
#                    print('++ INFO: Folder created [%s]' % out_dir)
#            out_path = osp.join(out_dir,'SIvsKNN_GroupLevel_{gm}_m{m}_Rdist.png'.format(m=str(m),gm=group_method))
#            plt.savefig(out_path,bbox_inches='tight')
#            plt.close()
#            if verbose:
#                print('++ INFO: Figure saved to disk [%s]' % out_path)
#    return None
   

# =============================================================================================
# =====                               DASHBOARD - UMAP KNN FIGURES
# =============================================================================================
def generate_UMAP_SIvsKnn_ScanLevel(si,sbj,figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200, init_method='spectral',min_dist=0.8, norm_methods=norm_methods):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for norm_method in norm_methods:
        for m in m_list:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            data_orig = si.loc[sbj,'Original',norm_method,init_method,min_dist,:,:,:,m,'Task',:].reset_index().drop(['Init','MinDist'],axis=1)
            data_nc   = si.loc[sbj,'Null_ConnRand',norm_method,init_method,min_dist,:,:,:,m,'Task',:].reset_index().drop(['Init','MinDist'],axis=1)
            data_np   = si.loc[sbj,'Null_PhaseRand',norm_method,init_method,min_dist,:,:,:,m,'Task',:].reset_index().drop(['Init','MinDist'],axis=1)
            g_orig = sns.lineplot(data=data_orig,y='SI',x='Knn',hue='Metric', style='Alpha',ax=ax, palette=ORIGINAL_PALETTE)
            g_nc   = sns.lineplot(data=data_nc,y='SI',x='Knn', hue='Metric', ax=ax, legend=False, palette=NULL_CONNRAND_PALETTE)
            g_np   = sns.lineplot(data=data_np,y='SI',x='Knn', hue='Metric', ax=ax, legend=False, palette=NULL_PHASERAND_PALETTE)
            ax.set_ylim(y_min,y_max)
            ax.set_xlim(x_min,x_max)
            ax.grid()
            ax.set_title('Scan-Level [{sbj}] | {m}D'.format(m=str(m),sbj=sbj), fontsize=HUGE_SIZE)
            ax.legend(loc='upper right', ncol=2)
            ax.set_ylabel('$SI_{%s}$' % 'Task')
            #ax.set_ylabel('SI [{target}]'.format(target='Task'))
            out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','UMAP')
            if not osp.exists(out_dir):
                os.makedirs(out_dir)
                if verbose:
                    print('++ INFO: Folder created [%s]' % out_dir)
            out_path = osp.join(out_dir,'SIvsKNN_ScanLevel_{sbj}_m{m}_{nm}_{target}.png'.format(m=str(m), target='Task',sbj=sbj,nm=norm_method))
            plt.savefig(out_path,bbox_inches='tight')
            if verbose:
                print('++ INFO: Figure saved to disk [%s]' % out_path)
            plt.close()
    return None

def generate_Avg_UMAP_SIvsKnn_ScanLevel(si,sbj_list,figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200, init_method='spectral',min_dist=0.8, norm_methods=norm_methods):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for norm_method in norm_methods:
        for m in m_list:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            data_orig = si.loc[sbj_list].loc[:,'Original'     , norm_method,init_method,min_dist,:,:,:,m,'Task',:].reset_index().drop(['Init','MinDist'],axis=1)
            data_nc   = si.loc[sbj_list].loc[:,'Null_ConnRand', norm_method,init_method,min_dist,:,:,:,m,'Task',:].reset_index().drop(['Init','MinDist'],axis=1)
            data_np   = si.loc[sbj_list].loc[:,'Null_PhaseRand',norm_method,init_method,min_dist,:,:,:,m,'Task',:].reset_index().drop(['Init','MinDist'],axis=1)
            sns.lineplot(data=data_orig, y='SI', x='Knn', hue='Metric', style='Alpha', hue_order=['correlation','cosine','euclidean'],ax=ax, palette=ORIGINAL_PALETTE)
            sns.lineplot(data=data_nc,   y='SI', x='Knn', hue='Metric', legend=False, ax=ax, palette=NULL_CONNRAND_PALETTE)
            sns.lineplot(data=data_np,   y='SI', x='Knn', hue='Metric', legend=False, ax=ax, palette=NULL_PHASERAND_PALETTE)
        
            ax.set_ylim(y_min,y_max)
            ax.set_xlim(x_min,x_max)
            ax.grid()
            ax.set_title('Scan-Level | {m}D'.format(m=str(m)), fontsize=HUGE_SIZE)
            ax.legend(loc='upper right', ncol=2)
            ax.set_ylabel('$SI_{%s}$' % 'Task')
            #ax.set_ylabel('SI [{target}]'.format(target='Task'))
            out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','UMAP')
            if not osp.exists(out_dir):
                os.makedirs(out_dir)
                if verbose:
                   print('++ INFO: Folder created [%s]' % out_dir)
            out_path = osp.join(out_dir,'SIvsKNN_ScanLevel_AVG_m{m}_{nm}_{target}.png'.format(m=str(m), target='Task',nm=norm_method))
            plt.savefig(out_path,bbox_inches='tight')
            if verbose:
                print('++ INFO: Figure saved to disk [%s]' % out_path)
            plt.close()
    return None
   
def generate_UMAP_SIvsKNN_GroupLevel(si,comb_methods=['ALL','Procrustes'],figsize=(10,5),target_m_tuples=[('Task',2),('Task',3),('Subject',2),('Subject',3)], 
                                     verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200,init_method='spectral',min_dist=0.8, norm_methods=norm_methods):
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
                data_orig = si.loc[group_method].loc['Original',norm_method,init_method,min_dist,:,:,:,m,target,:]
                data_nc = si.loc[group_method].loc['Null_ConnRand',norm_method,init_method,min_dist,:,:,:,m,target,:]
                data_np = si.loc[group_method].loc['Null_PhaseRand',norm_method,init_method,min_dist,:,:,:,m,target,:]
                #sns.lineplot(data=data_nc,  y='SI',x='Knn', hue='Metric', hue_order=['correlation','cosine','euclidean'],legend=False,ax=ax, palette=NULL_CONNRAND_PALETTE)
                #sns.lineplot(data=data_np,  y='SI',x='Knn', hue='Metric', hue_order=['correlation','cosine','euclidean'],legend=False,ax=ax, palette=NULL_PHASERAND_PALETTE)
                sns.lineplot(data=data_nc,  y='SI',x='Knn', legend=False,ax=ax, color=NULL_CONNRAND_PALETTE[0])
                sns.lineplot(data=data_np,  y='SI',x='Knn', legend=False,ax=ax, color=NULL_PHASERAND_PALETTE[0])
                sns.lineplot(data=data_orig,y='SI',x='Knn', hue='Metric', hue_order=['correlation','cosine','euclidean'],             ax=ax, palette=ORIGINAL_PALETTE)
                ax.set_ylim(y_min,y_max)
                ax.set_xlim(x_min,x_max)
                ax.grid()
                ax.set_title('Group-Level [{gm}] | {m}D'.format(m=str(m),gm=group_method_2_label[group_method]), fontsize=HUGE_SIZE)
                ax.legend(loc='upper right', ncol=2)
                ax.set_ylabel('$SI_{%s}$' % target)
                #ax.set_ylabel('SI [{target}]'.format(target=target))
                out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','UMAP')
                if not osp.exists(out_dir):
                    os.makedirs(out_dir)
                    if verbose:
                        print('++ INFO: Folder created [%s]' % out_dir)
                out_path = osp.join(out_dir,'SIvsKNN_GroupLevel_{gm}_m{m}_{nm}_{target}.png'.format(m=str(m), target=target, gm=group_method, nm=norm_method))
                plt.savefig(out_path,bbox_inches='tight')
                if verbose:
                    print('++ INFO: Figure saved to disk [%s]' % out_path)
                plt.close()
    return None

# =============================================================================================
# =====                               DASHBOARD - TSNE KNN FIGURES
# =============================================================================================
def generate_TSNE_SIvsKnn_ScanLevel(si,sbj,figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200, init_method='pca', norm_methods=norm_methods):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for norm_method in norm_methods:
        for m in m_list:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            data_orig = si.loc[sbj,'Original',norm_method,:,:,m,:,init_method,'Task',:].reset_index().drop(['Init'],axis=1)
            data_nc   = si.loc[sbj,'Null_ConnRand',norm_method,:,:,m,:,init_method,'Task',:].reset_index().drop(['Init'],axis=1)
            data_np   = si.loc[sbj,'Null_PhaseRand',norm_method,:,:,m,:,init_method,'Task',:].reset_index().drop(['Init'],axis=1)
            g_orig = sns.lineplot(data=data_orig,y='SI',x='PP',hue='Metric', style='Alpha',ax=ax, palette=ORIGINAL_PALETTE)
            g_nc   = sns.lineplot(data=data_nc,y='SI',x='PP', ax=ax,  legend=False, palette=NULL_CONNRAND_PALETTE)
            g_nc   = sns.lineplot(data=data_np,y='SI',x='PP', ax=ax,  legend=False, palette=NULL_PHASERAND_PALETTE)
            ax.set_ylim(y_min,y_max)
            ax.set_xlim(x_min,x_max)
            ax.grid()
            ax.set_title('Scan-Level [{sbj}] | {m}D'.format(m=str(m),sbj=sbj), fontsize=HUGE_SIZE)
            ax.legend(loc='upper right', ncol=2)
            ax.set_ylabel('$SI_{%s}$' % 'Task')
            #ax.set_ylabel('SI [{target}]'.format(target='Task'))
            out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','TSNE')
            if not osp.exists(out_dir):
                os.makedirs(out_dir)
                if verbose:
                    print('++ INFO: Folder created [%s]' % out_dir)
            out_path = osp.join(out_dir,'SIvsPP_ScanLevel_{sbj}_m{m}_{nm}_{target}.png'.format(m=str(m), target='Task',sbj=sbj,nm=norm_method))
            plt.savefig(out_path,bbox_inches='tight')
            if verbose:
                print('++ INFO: Figure saved to disk [%s]' % out_path)
            plt.close()
    return None

def generate_Avg_TSNE_SIvsKnn_ScanLevel(si,sbj_list,figsize=(10,5),m_list=[2,3], verbose=False, y_min=-0.2, y_max=0.85,x_min=5,x_max=200, init_method='pca', norm_methods=norm_methods):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=HUGE_SIZE)    # fontsize of the figure title
    for norm_method in norm_methods:
        for m in m_list:
            fig, ax = plt.subplots(1,1,figsize=figsize)
            data_orig = si.loc[sbj_list].loc[:,'Original',       norm_method, :,:,m,:,init_method,'Task',:].reset_index().drop(['Init'],axis=1)
            data_nc   = si.loc[sbj_list].loc[:,'Null_ConnRand',  norm_method, :,:,m,:,init_method,'Task',:].reset_index().drop(['Init'],axis=1)
            data_np   = si.loc[sbj_list].loc[:,'Null_PhaseRand', norm_method, :,:,m,:,init_method,'Task',:].reset_index().drop(['Init'],axis=1)
            sns.lineplot(data=data_orig, y='SI', x='PP', hue='Metric', style='Alpha', hue_order=['correlation','cosine','euclidean'],ax=ax, palette=ORIGINAL_PALETTE)
            sns.lineplot(data=data_nc,   y='SI', x='PP', hue='Metric', legend=False, ax=ax, palette=NULL_CONNRAND_PALETTE)
            sns.lineplot(data=data_np,   y='SI', x='PP', hue='Metric', legend=False, ax=ax, palette=NULL_PHASERAND_PALETTE)
        
            ax.set_ylim(y_min,y_max)
            ax.set_xlim(x_min,x_max)
            ax.grid()
            ax.set_title('Scan-Level | {m}D'.format(m=str(m)), fontsize=HUGE_SIZE)
            ax.legend(loc='upper right', ncol=2)
            ax.set_ylabel('$SI_{%s}$' % 'Task')
            #ax.set_ylabel('SI [{target}]'.format(target='Task'))
            out_dir = osp.join(PRJ_DIR,'Dashboard','Figures','TSNE')
            if not osp.exists(out_dir):
                os.makedirs(out_dir)
                if verbose:
                   print('++ INFO: Folder created [%s]' % out_dir)
            out_path = osp.join(out_dir,'SIvsPP_ScanLevel_AVG_m{m}_{nm}_{target}.png'.format(m=str(m), target='Task',nm=norm_method))
            plt.savefig(out_path,bbox_inches='tight')
            if verbose:
                print('++ INFO: Figure saved to disk [%s]' % out_path)
            plt.close()
    return None

   
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
                data_orig = si.loc[group_method].loc['Original',norm_method,:,:,:m,:,init_method,target]
                data_nc = si.loc[group_method].loc['Null_ConnRand',norm_method,:,:,:m,:,init_method,target]
                data_np = si.loc[group_method].loc['Null_PhaseRand',norm_method,:,:,:m,:,init_method,target]

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
                plt.close()