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
  const env_spec = ['https://cdn.holoviz.org/panel/wheels/bokeh-3.6.2-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.6.0/dist/wheels/panel-1.6.0-py3-none-any.whl', 'pyodide-http==0.2.1', 'hvplot', 'numpy', 'pandas', 'plotly', 'scipy']
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
  \nimport asyncio\n\nfrom panel.io.pyodide import init_doc, write_doc\n\ninit_doc()\n\n# ---\n# jupyter:\n#   jupytext:\n#     formats: ipynb,py:light\n#     text_representation:\n#       extension: .py\n#       format_name: light\n#       format_version: '1.5'\n#       jupytext_version: 1.15.2\n#   kernelspec:\n#     display_name: opentsne_panel14\n#     language: python\n#     name: opentsne_panel14\n# ---\n\n# + active=""\n# cd /data/SFIMJGC_HCP7T/manifold_learning_fmri/Notebooks\n# panel convert GUI_Embeddings_toDeploy.py --to pyodide-worker --out ../docs/ --pwa --title manifold_fmri\n# -\n\nimport panel as pn\nimport numpy as np\nimport os.path as osp\nimport pandas as pd\nfrom scipy.stats import zscore\n#from matplotlib.colors import rgb2hex\n#import matplotlib\nimport hvplot.pandas\nimport plotly\npn.extension('plotly')\nimport plotly.express as px\n\n# So far we are working with these values of wls and wss across the whole manuscript\nwls = 45\nwss = 1.5\nmin_dist = 0.8\n\n# +\nDATA_URL = 'https://raw.githubusercontent.com/nimh-sfim/manifold_learning_fmri_demo_data/master/data/'\nPRJ_DIR  = '/data/SFIMJGC_HCP7T/manifold_learning_fmri'\n\n# Available scans\navail_scans_dict = {'Scan 1':'SBJ06', 'Scan 2':'SBJ07'}\n\n# Available Data Scenarios\ninput_data_dict = {'Real Data':'Original','Connectivity Randomization':'Null_ConnRand','Phase Randomization':'Null_PhaseRand'}\n\n# Normalization Options\nnormalization_dict = {'Do not normalize':'asis','Z-score':'zscored'}\n\n# Colormaps\n#sbj_cmap_list = [rgb2hex(c) for c in matplotlib.colormaps['tab20'].colors]\n# Hard coded below to avoid importing matplotlib\nsbj_cmap_list = ['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5']\nsbj_cmap = {v:sbj_cmap_list[i] for i,v in enumerate(avail_scans_dict.values())}\ntask_cmap = {'REST': 'gray', 'BACK': 'blue',   'VIDE':  '#F4D03F',  'MATH': 'green', 'XXXX': 'pink'}\n\n# Laplacian Eigenmap related options\nle_dist_metrics = {'Euclidean Distance':'euclidean','Correlation Distance':'correlation','Cosine Distance':'cosine'}\nle_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)][::5]\nle_ms           = [2,3,5,10,15]\n\n# UMAP related options\numap_dist_metrics = le_dist_metrics\numap_knns         = [int(i) for i in np.linspace(start=5, stop=200, num=40)][::5]\numap_ms           = le_ms\numap_alphas       = [0.01, 0.1, 1.0]\numap_inits        = ['spectral']\n\n# T-SNE related options\ntsne_dist_metrics = le_dist_metrics\ntsne_pps          = [int(i) for i in np.linspace(start=5, stop=100, num=20)] + [125, 150, 175, 200]\ntsne_ms           = le_ms\ntsne_alphas       = [10, 50, 75, 100, 200, 500, 1000]\ntsne_inits        = ['pca']\n# -\n\n# ***\n# ### Functions from utils.plotting\n\ncamera = dict( up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=1)) \n\n\ndef plot_2d_scatter(data,x,y,c,cmap=task_cmap, show_frame=False, s=2, alpha=0.3, toolbar=None, \n                    legend=False, xaxis=False, xlabel='', yaxis=False, ylabel='', frame_width=300, shared_axes=False):\n    plot = data.hvplot.scatter(x=x,y=y,c=c, cmap=cmap, \n                            aspect='square', s=s, alpha=alpha, \n                            legend=legend, xaxis=xaxis, \n                            yaxis=yaxis, frame_width=frame_width, shared_axes=shared_axes).opts(toolbar=toolbar, show_frame=show_frame, tools=[])\n    return plot\n\n\ndef plot_3d_scatter(data,x,y,z,c,cmap,s=2,width=400, height=400, ax_range=[-.005,.005],nticks=4):\n    fig = px.scatter_3d(data,\n                        x=x,y=y,z=z, \n                        width=width, height=height, \n                        opacity=0.3, color=c,color_discrete_sequence=cmap)\n    fig.update_layout(showlegend=False, \n                          font_color='white');\n    scene_extra_confs = dict(\n        xaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),\n        yaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'),\n        zaxis = dict(nticks=nticks, range=ax_range, gridcolor="black", showbackground=True, zerolinecolor="black",backgroundcolor='rgb(230,230,230)'))\n    fig.update_layout(scene_camera=camera, scene=scene_extra_confs, scene_aspectmode='cube',margin=dict(l=2, r=2, b=0, t=0, pad=0))\n    fig.update_traces(marker_size = s)\n    return fig\n\n\n# ***\n# ### Functions from utils.io\n\ndef load_single_le(sbj,input_data,scenario,dist,knn,m,wls=45,wss=1.5, drop_xxxx=True, show_path=False):\n    path = osp.join(DATA_URL,'embeddings',sbj,'LE',input_data,\n                    '{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(sbj=sbj,scenario=scenario,wls=str(int(wls)).zfill(3),wss=str(wss),\n                                                                                                        dist=dist,knn=str(knn).zfill(4),m=str(m).zfill(4)))\n    try:\n        aux = pd.read_pickle(path)\n    except:\n        return None\n    if drop_xxxx:\n        if type(aux.index) is pd.MultiIndex:\n            aux = aux.drop('XXXX', level='Window Name')\n        else:\n            aux = aux.drop('XXXX',axis=0)\n    return aux\n\n\n# ***\n# # Main Dashboard Panel: Configuration Options\n\nsbj_select      = pn.widgets.Select(name='fMRI Scan',     options=avail_scans_dict,  width=150, description='Select the scan you want to explore')\ninput_select    = pn.widgets.Select(name='Scenario',      options=input_data_dict,    width=150, description='Select original data or null data (phase or connection randomized)')\nscenario_select = pn.widgets.Select(name='Normalization', options=normalization_dict, width=150,description='Select whether or not to normalize data prior to embedding estimation')\n\n# ***\n# # Laplacian Eigenmaps\n\n# #### 1. Load Silhouette Index for LE\n\nsi_LE_URL = osp.join(DATA_URL,'sil_index','si_LE.pkl')\nsi_LE = pd.read_pickle(si_LE_URL)\n\n# #### 3. LE Tab Elements\n\nle_m_select     = pn.widgets.Select(name='M',   options=le_ms, value=le_ms[-1], width=150, description='Number of dimensions used for computing the left-most embedding (independently of M, the plot will only show the first three dimensions)')\nle_knn_select   = pn.widgets.Select(name='Knn', options=le_knns,         value=le_knns[0], width=150, description='Neighborhood Size for Laplacian Embeddings')\nle_dist_select  = pn.widgets.Select(name='Distance Metric', options=le_dist_metrics, width=150,description='Distance metric used when computing Laplacian Embeddings')\nle_drop_xxxx    = pn.widgets.Checkbox(name='Drop Mixed Windows?', width=150)\nle_conf_box     = pn.WidgetBox(le_dist_select,le_knn_select,le_m_select,le_drop_xxxx)\n\n\ndef plot_LE_scats(group_type,input_data,scenario,dist,knn,m,color_col,plot_2d_toolbar,drop_xxxx):\n    plots = None\n    aux_2d, aux_3d, aux_Md = None, None, None\n    #if group_type in ['Procrustes','ALL']:\n    #    sitable_2d, sitable_3d, sitable_Md = pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Subject','Task'],name='Target'),columns=['SI']),width=150),pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Subject','Task'],name='Target'),columns=['SI']),width=150),pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Subject','Task'],name='Target'),columns=['SI']),width=150)\n    #else:\n    #    sitable_2d, sitable_3d, sitable_Md = pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Task'],name='Target'),columns=['SI']),width=150),\n    #                                         pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Task'],name='Target'),columns=['SI']),width=150),\n    #                                         pn.pane.DataFrame(pd.DataFrame(index=pd.Index(['Task'],name='Target'),columns=['SI']),width=150)\n    # Load all necessary embeddings\n    # =============================\n    if m == 2:\n        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)\n    elif m == 3:\n        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)\n        aux_3d = load_single_le(group_type,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)\n    else:\n        aux_2d = load_single_le(group_type,input_data,scenario,dist,knn,2,drop_xxxx=drop_xxxx)\n        aux_3d = load_single_le(group_type,input_data,scenario,dist,knn,3,drop_xxxx=drop_xxxx)\n        aux_Md = load_single_le(group_type,input_data,scenario,dist,knn,m,drop_xxxx=drop_xxxx)\n    # Preprare Embeddings\n    # ===================\n    if not (aux_2d is None):\n        aux_2d = aux_2d.apply(zscore)\n        aux_2d = aux_2d.reset_index()\n        \n    if not (aux_3d is None):\n         aux_3d = aux_3d.apply(zscore)\n         aux_3d = aux_3d.reset_index()\n\n    if not (aux_Md is None):\n         aux_Md = aux_Md.apply(zscore)\n         aux_Md = aux_Md.reset_index()\n    # Prepare SI Tables\n    # =================\n    if (group_type,input_data,scenario,dist,knn,2) in si_LE.index:\n        sitable_2d = pn.pane.Markdown("<p align='center' style='font-size:20px'><a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>Silhouette Score</a> = %.2f </p>" % si_LE.loc[group_type,input_data,scenario,dist,knn,2,'Task']['SI'].item(), width=350, align='center')\n    if (group_type,input_data,scenario,dist,knn,3) in si_LE.index:\n        sitable_3d = pn.pane.Markdown("<p align='center' style='font-size:20px'><a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>Silhouette Score</a> = %.2f </p>" % si_LE.loc[group_type,input_data,scenario,dist,knn,3,'Task']['SI'].item(), width=350, align='center')\n    if (group_type,input_data,scenario,dist,knn,m) in si_LE.index:\n        sitable_Md = pn.pane.Markdown("<p align='center' style='font-size:20px'><a href='https://en.wikipedia.org/wiki/Silhouette_(clustering)' target='_blank'>Silhouette Score</a> = %.2f </p>" % si_LE.loc[group_type,input_data,scenario,dist,knn,m,'Task']['SI'].item(), width=350, align='center')\n    # Prepare Color-scales\n    # ====================\n    if color_col == 'Subject':\n        cmap_2d = sbj_cmap\n        cmap_3d = sbj_cmap_list\n    else:\n        cmap_2d = task_cmap\n        if not(aux_3d is None):\n            cmap_3d = [task_cmap[t] for t in aux_3d['Window Name'].unique()]\n    # Plotting\n    # ========\n    if (not (aux_2d is None)) & (aux_3d is None):\n        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),sitable_2d],ncols=1)\n    if (not (aux_2d is None)) & (not (aux_3d is None)) & (aux_Md is None):\n        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),\n                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),\n                             sitable_2d, sitable_3d],ncols=2)\n    if (not (aux_2d is None)) & (not (aux_3d is None)) & (not (aux_Md is None)):\n        plots = pn.GridBox(*[plot_2d_scatter(aux_2d,x='LE001',y='LE002',c=color_col, cmap=cmap_2d, s=10, toolbar=plot_2d_toolbar),\n                             plot_3d_scatter(aux_3d,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),\n                             plot_3d_scatter(aux_Md,x='LE001',y='LE002',z='LE003',c=color_col, cmap=cmap_3d,s=3, ax_range=[-2,2]),\n                             sitable_2d,sitable_3d,sitable_Md],ncols=3) \n    return plots\n\n\n@pn.depends(sbj_select,input_select,scenario_select,le_dist_select,le_knn_select,le_m_select,le_drop_xxxx)\ndef plot_LE_Scan_scats(sbj,input_data,scenario,dist,knn,m, drop_xxxx):\n    return plot_LE_scats(sbj,input_data,scenario,dist,knn,m,'Window Name','above',drop_xxxx)\n\n\nle_config_card                = pn.Column(le_conf_box)\nle_embs_scan_card             = pn.layout.Card(plot_LE_Scan_scats,title='Laplacian Eigenmaps - Single fMRI Scan', width=1200, header_background='#0072B5', header_color='#ffffff')\nle_embs_col = pn.Column(le_embs_scan_card)\n\nle_tab=pn.Row(le_config_card,le_embs_col)\n\n# ***\n# # UMAP\n# #### 1. Load Silhouette Index for UMAP\n#\n# Hyper-parameter space: 3 Inputs * 2 Norm Approach * 8 m * 3 dist * x knns * 3 alphas = \n# * "Concat + UMAP": 17280 entries\n# * "UMAP + Procrustes": 17280 entries\n# * Single-Scan Level: 345600 entries\n\n# ***\n# # TSNE\n# #### 1. Load Silhouette Index for TSNE\n\n# ***\n\n# +\n# Instantiate the template with widgets displayed in the sidebar\ntemplate = pn.template.FastListTemplate(\n    title="Manifold Learning for time-varying functional connectivity",\n    sidebar=[sbj_select,input_select,scenario_select],\n    sidebar_width=200,\n    theme_toggle=False\n    \n)\n# -\n\nintro_text = pn.pane.Markdown("""\nThis dashbaord allows you to explore time-vayring fMRI data embedded using three state-of-the-art techniques. It is a companion to a publications in Frontiers in Neuroscience that you can find [here](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1134012/full).\n""", width=1000)\ntemplate.main.append(pn.Column(intro_text,pn.Tabs(('Laplacian Eigenmaps',le_tab)))) #,('T-SNE',tsne_tab),('UMAP',umap_tab))))\n\ntemplate.servable()\n#dashboard = template.show(port=port_tunnel)\n\n# +\n# import os\n# port_tunnel = int(os.environ['PORT2'])\n# print('++ INFO: Second Port available: %d' % port_tunnel)\n# dashboard = template.show(port=port_tunnel)\n# -\n\n\n\n\nawait write_doc()
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