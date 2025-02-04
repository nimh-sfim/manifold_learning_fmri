importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/wheels/bokeh-3.6.2-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.6.0/dist/wheels/panel-1.6.0-py3-none-any.whl', 'pyodide-http==0.2.1', 'hvplot', 'matplotlib', 'numpy', 'pandas', 'plotly', 'scipy']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  \nimport asyncio\n\nfrom panel.io.pyodide import init_doc, write_doc\n\ninit_doc()\n\n# ---\n# jupyter:\n#   jupytext:\n#     formats: ipynb,py:light\n#     text_representation:\n#       extension: .py\n#       format_name: light\n#       format_version: '1.5'\n#       jupytext_version: 1.15.2\n#   kernelspec:\n#     display_name: opentsne_panel14\n#     language: python\n#     name: opentsne_panel14\n# ---\n\n# +\nimport panel as pn\nimport numpy as np\nimport os.path as osp\nimport pandas as pd\nfrom scipy.stats import zscore\nfrom matplotlib.colors import rgb2hex\nimport matplotlib\n\nimport hvplot.pandas\nimport plotly.express as px\n# -\n\n# So far we are working with these values of wls and wss across the whole manuscript\nwls = 45\nwss = 1.5\nmin_dist = 0.8\n\n# +\nDATA_URL = 'https://github.com/nimh-sfim/manifold_learning_fmri/raw/refs/heads/main/Data/Dashboard/GUI_Embeddings'\nPRJ_DIR          = '/data/SFIMJGC_HCP7T/manifold_learning_fmri'\n\nPNAS2015_subject_list   = ['SBJ06', 'SBJ07']#, 'SBJ08', 'SBJ09', 'SBJ10', 'SBJ11', 'SBJ12', 'SBJ13', 'SBJ16', 'SBJ17', 'SBJ18', 'SBJ19', 'SBJ20', 'SBJ21', 'SBJ22', 'SBJ23', 'SBJ24', 'SBJ25', 'SBJ26', 'SBJ27']\n\n# Colormaps\nsbj_cmap_list = [rgb2hex(c) for c in matplotlib.colormaps['tab20'].colors]\nsbj_cmap_dict = {PNAS2015_subject_list[i]:sbj_cmap_list[i] for i in range(len(PNAS2015_subject_list))}\ntask_cmap = {'REST': 'gray', 'BACK': 'blue',   'VIDE':  '#F4D03F',  'MATH': 'green', 'XXXX': 'pink'}\n\n# Laplacian Eigenmap related options\nle_dist_metrics = ['euclidean','correlation','cosine']\nle_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)][::5]\nle_ms           = [2,3,5,10,15] #,20,25,30]\n\n# UMAP related options\numap_dist_metrics = ['euclidean','correlation','cosine']\numap_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)]\numap_ms           = [2,3,5,10,15] #,20,25,30]\numap_alphas       = [0.01, 0.1, 1.0]\numap_inits        = ['spectral']\n\n# T-SNE related options\ntsne_dist_metrics = ['euclidean','correlation','cosine']\ntsne_pps          = [int(i) for i in np.linspace(start=5, stop=100, num=20)] + [125, 150, 175, 200]\ntsne_ms           = [2,3,5,10,15] #,20,25,30]\ntsne_alphas       = [10, 50, 75, 100, 200, 500, 1000]\ntsne_inits        = ['pca']\n\n# Additional lists of options\ninput_datas  = ['Original','Null_ConnRand','Null_PhaseRand']\nnorm_methods = ['asis','zscored']\n# -\n\n# ***\n# ### Functions from utils.plotting\n\ncamera = dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=1)) \n\n\ndef plot_2d_scatter(data,x,y,c,cmap=task_cmap, show_frame=False, s=2, alpha=0.3, toolbar=None, \n                    legend=False, xaxis=False, xlabel='', yaxis=False, ylabel='', frame_width=250, shared_axes=False):\n    plot = data.hvplot.scatter(x=x,y=y,c=c, cmap=cmap, \n                            aspect='square', s=s, alpha=alpha, \n                            legend=legend, xaxis=xaxis, \n                            yaxis=yaxis, frame_width=frame_width, shared_axes=shared_axes).opts(toolbar=toolbar, show_frame=show_frame, tools=[])\n    return plot\n\n\ndef plot_3d_scatter(data,x,y,z,c,cmap,s=2,width=250, height=250, ax_range=[-.005,.005],nticks=4):\n    fig = px.scatter_3d(data,\n                        x=x,y=y,z=z, \n                        width=width, height=height, \n                        opacity=0.3, color=c,color_discrete_sequence=cmap)\n    fig.update_layout(showlegend=False, \n                          font_color='white');\n    scene_extra_confs = dict(\n        xaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),\n        yaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),\n        zaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))\n    fig.update_layout(scene_camera=camera, scene=scene_extra_confs, scene_aspectmode='cube',margin=dict(l=2, r=2, b=0, t=0, pad=0))\n    fig.update_traces(marker_size = s)\n    return fig\n\n\n# ***\n# ### Functions from utils.io\n\ndef load_single_le(sbj,input_data,scenario,dist,knn,m,wls=45,wss=1.5, drop_xxxx=True, show_path=False):\n    path = osp.join(DATA_URL,sbj,'LE',input_data,\n                    '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(sbj=sbj,scenario=scenario,wls=str(int(wls)).zfill(3),wss=str(wss),\n                                                                                                        dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4)))\n    try:\n        aux = pd.read_pickle(path)\n    except:\n        return None\n    if drop_xxxx:\n        if type(aux.index) is pd.MultiIndex:\n            aux = aux.drop('XXXX', level='Window Name')\n        else:\n            aux = aux.drop('XXXX',axis=0)\n    return aux\n\n\n# ***\n# # Main Dashboard Panel: Configuration Options\n\nsbj_select = pn.widgets.Select(name='Scan', options=PNAS2015_subject_list, value=PNAS2015_subject_list[0], width=150, description='Select the scan you want to explore in the single-scan section.')\ninput_select    = pn.widgets.Select(name='Scenario', options=['Original','Null_ConnRand','Null_PhaseRand'], value='Original', width=150, description='Use original data or randomize data (phase or connection randomized)')\nscenario_select = pn.widgets.Select(name='Normalization',           options=['asis','zscored'], value='asis', width=150,description='Select asis to indicate no normalization, or z-score if you want to normalize the data')\nplot2d_toolbar_select = pn.widgets.Select(name='2D Toolbar', options=['above', 'below', 'left', 'right', 'disable'], value='disable', width=150, description='Sometimes toolbars get on the way. Here you can select its location or to disable it') \n\n# ***\n# # Laplacian Eigenmaps\n\n# #### 1. Load Silhouette Index for LE\n\n#si_LE_URL = osp.join(DATA_URL,'si_LE.pkl')\n#si_LE = pd.read_pickle(si_LE_URL)\nsi_LE_URL = osp.join(DATA_URL,'si_LE.csv')\nsi_LE = pd.read_csv(si_LE_URL, index_col=[0,1,2,3,4,5,6])\n\n# #### 3. LE Tab Elements\n\nle_m_select     = pn.widgets.Select(name='M',               options=[2,3,5,10,15,20,25,30],         value=5, width=150, description='Number of dimensions used for computing the embedding. This only affects the right-most 3D plot, as the 2D (left) and 3D (middle) embedding are always computed using 2 and 3 dimensions respectively.')\nle_knn_select   = pn.widgets.Select(name='Knn',             options=le_knns,         value=le_knns[0], width=150, description='Neighborhood Size for Laplacian Embeddings')\nle_dist_select  = pn.widgets.Select(name='Distance Metric', options=le_dist_metrics, value=le_dist_metrics[0], width=150,description='Distance metric used when computing Laplacian Embeddings')\nle_grcc_col_sel = pn.widgets.Select(name='[G-CC] Color By:', options=['Window Name','Subject'], value='Window Name', width=150, description='Color points in Group-level concatenated embedding according to task or scan membership.')\nle_grpt_col_sel = pn.widgets.Select(name='[G-PT] Color By:', options=['Window Name','Subject'], value='Window Name', width=150, description='Color points in Group-level Procrustes embedding according to task or scan membership.')\nle_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows?', width=150)\nle_conf_box     = pn.WidgetBox(le_dist_select,le_knn_select,le_m_select,le_grcc_col_sel,le_grpt_col_sel,le_drop_xxxx)\n\n\ndef plot_LE_scats(group_type,input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):\n    plots = None\n    aux_2d, aux_3d, aux_Md = None, None, None\n    if group_type in ['Procrustes','ALL']:\n        sitable_2d, sitable_3d, sitable_Md = pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Subject','Task'],name='Target'),columns=['SI']),width=150),pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Subject','Task'],name='Target'),columns=['SI']),width=150),pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Subject','Task'],name='Target'),columns=['SI']),width=150)\n    else:\n        sitable_2d, sitable_3d, sitable_Md = pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Task'],name='Target'),columns=['SI']),width=150),pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Task'],name='Target'),columns=['SI']),width=150),pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Task'],name='Target'),columns=['SI']),width=150)\n    # Load all necessary embeddings\n    # =============================\n    if m == 2:\n        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)\n    elif m == 3:\n        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)\n        aux_3d = load_single_le(group_type,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)\n    else:\n        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)\n        aux_3d = load_single_le(group_type,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)\n        aux_Md = load_single_le(group_type,input_data,scenario,dist,knn,m,drop_xxxx=drop_xxxx)\n    # Preprare Embeddings\n    # ===================\n    if not (aux_2d is None):\n        aux_2d = aux_2d.apply(zscore)\n        aux_2d = aux_2d.reset_index()\n    #    aux_2d.set_index('Window Name', inplace=True)\n    #    aux_2d = aux_2d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)\n    #    aux_2d = aux_2d.reset_index()\n    \n    if not (aux_3d is None):\n         aux_3d = aux_3d.apply(zscore)\n         aux_3d = aux_3d.reset_index()\n     #    aux_3d.set_index('Window Name', inplace=True)\n     #    aux_3d = aux_3d.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)\n     #    aux_3d = aux_3d.reset_index()\n\n    if not (aux_Md is None):\n         aux_Md = aux_Md.apply(zscore)\n         aux_Md = aux_Md.reset_index()\n      #   aux_Md.set_index('Window Name', inplace=True)\n      #   aux_Md = aux_Md.sort_index(level='Window Name',ascending=False) # So Inbetween are plotted in the back (for clarity)\n      #   aux_Md = aux_Md.reset_index()\n\n    # Prepare SI Tables\n    # =================\n    if (group_type,input_data,scenario,dist,knn,2) in si_LE.index:\n        sitable_2d = pn.pane.DataFrame(si_LE.loc[group_type,input_data,scenario,dist,knn,2].round(2),width=150)\n    if (group_type,input_data,scenario,dist,knn,3) in si_LE.index:\n        sitable_3d = pn.pane.DataFrame(si_LE.loc[group_type,input_data,scenario,dist,knn,3].round(2),width=150)\n    if (group_type,input_data,scenario,dist,knn,m) in si_LE.index:\n        sitable_Md = pn.pane.DataFrame(si_LE.loc[group_type,input_data,scenario,dist,knn,m].round(2),width=150)\n    # Prepare Color-scales\n    # ====================\n    if color_col == 'Subject':\n        cmap_2d = sbj_cmap_dict\n        cmap_3d = sbj_cmap_list\n    else:\n        cmap_2d = task_cmap\n        if not(aux_3d is None):\n            cmap_3d = [task_cmap[t] for t in aux_3d['Window Name'].unique()]\n    # Plotting\n    # ========\n    if (not (aux_2d is None)) & (aux_3d is None):\n        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),sitable_2d],ncols=1)\n    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):\n        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),\n                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),\n                             sitable_2d, sitable_3d],ncols=2)\n    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):\n        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),\n                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),\n                             plot_3d_scatter(aux_Md,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),\n                             sitable_2d,sitable_3d,sitable_Md],ncols=3) \n    return plots\n\n\n@pn.depends(input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,le_grcc_col_sel,plot2d_toolbar_select,le_drop_xxxx)\ndef plot_LE_Group_Concat_scats(input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):\n    return plot_LE_scats('ALL',input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx)\n@pn.depends(input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,le_grpt_col_sel,plot2d_toolbar_select,le_drop_xxxx)\ndef plot_LE_Group_Procrustes_scats(input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):\n    return plot_LE_scats('Procrustes',input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx)\n@pn.depends(sbj_select,input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,plot2d_toolbar_select,le_drop_xxxx)\ndef plot_LE_Scan_scats(sbj,input_data,scenario,dist,knn,m,plot_2d_toolbar, drop_xxxx):\n    return plot_LE_scats(sbj,input_data,scenario,dist,knn,m,'Window Name',plot_2d_toolbar,drop_xxxx)\n\n\nle_config_card                = pn.Column(le_conf_box)\nle_embs_scan_card             = pn.layout.Card(plot_LE_Scan_scats,title='Scatter Plots - One Scan', width=825)\nle_embs_group_concat_card     = pn.layout.Card(plot_LE_Group_Concat_scats,title='Scatter Plots - Group Concatenation', width=825)\nle_embs_group_procrustes_card = pn.layout.Card(plot_LE_Group_Procrustes_scats,title='Scatter Plots - Procrustes', width=825)\nle_embs_col = pn.Column(le_embs_scan_card ,le_embs_group_concat_card,le_embs_group_procrustes_card)\n\nle_tab=pn.Row(le_config_card,le_embs_col)\n\n# ***\n# # UMAP\n# #### 1. Load Silhouette Index for UMAP\n#\n# Hyper-parameter space: 3 Inputs * 2 Norm Approach * 8 m * 3 dist * x knns * 3 alphas = \n# * "Concat + UMAP": 17280 entries\n# * "UMAP + Procrustes": 17280 entries\n# * Single-Scan Level: 345600 entries\n\n# ***\n# # TSNE\n# #### 1. Load Silhouette Index for TSNE\n\n# ***\n\n# Instantiate the template with widgets displayed in the sidebar\ntemplate = pn.template.FastListTemplate(\n    title="Manifold Learning for time-varying functional connectivity",\n    sidebar=[sbj_select,input_select,scenario_select,plot2d_toolbar_select],\n    header_color='#ffffff',\n    sidebar_width=200\n)\n\nimport os\ncwd = os.getcwd()\nprint('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')\nprint(cwd)\nprint('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')\n\n\nintro_text = pn.pane.Markdown("""\nThis dashbaord allows you to explore time-vayring fMRI data embedded using three state-of-the-art techniques. It is a companion to a publications in Frontiers in Neuroscience that you can find [here](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1134012/full).\n""", width=1000)\ntemplate.main.append(pn.Column(intro_text,pn.Tabs(('Laplacian Eigenmaps',le_tab)))) #,('T-SNE',tsne_tab),('UMAP',umap_tab))))\n\ntemplate.servable()\n#dashboard = template.show(port=port_tunnel)\n\n# +\n# import os\n# port_tunnel = int(os.environ['PORT2'])\n# print('++ INFO: Second Port available: %d' % port_tunnel)\n# dashboard = template.show(port=port_tunnel)\n# -\n\n\n\n\nawait write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    from panel.io.pyodide import _convert_json_patch
    state.curdoc.apply_json_patch(_convert_json_patch(patch), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()