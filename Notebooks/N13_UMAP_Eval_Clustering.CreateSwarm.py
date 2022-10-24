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
#     display_name: opentsne
#     language: python
#     name: opentsne
# ---

# +
from utils.basics import umap_dist_metrics, umap_knns, umap_ms, umap_alphas
from utils.basics import PRJ_DIR, PNAS2015_subject_list
from utils.basics import input_datas, norm_methods
from datetime import datetime

import os.path as osp
import os
import getpass
# -

wls = 45
wss = 1.5
min_dist = 0.8

# ### Scan-level results

umap_ms = [2,3]
init_method = 'spectral'

print('++ INFO: Distance Metrics: %s' % str(umap_dist_metrics))
print('++ INFO: Knns:             %s' % str(umap_knns))
print('++ INFO: Ms:               %s' % str(umap_ms))
print('++ INFO: Learning Rates:   %s' % str(umap_alphas))

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N13_UMAP_Eval_Clustering_Scans.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N13_UMAP_Eval_Clustering_Scans.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)
# -

# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
# Insert comment line with SWARM command
swarm_file.write('#swarm -J UMAP_Scans_SI -f {swarm_path} -b 57 -g 16 -t 4 --time 00:03:00 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
for input_data in input_datas:
    for norm_method in norm_methods:
        for dist in umap_dist_metrics:
            for knn in umap_knns:
                for m in umap_ms:
                    for alpha in umap_alphas:
                        for sbj in PNAS2015_subject_list:
                            num_iters +=1
                            input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.pkl'.format(norm_method=norm_method,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   alpha=str(alpha)))
                            output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.SI.pkl'.format(norm_method=norm_method, init_method=init_method,
                                                                                                                                                   sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                            if not osp.exists(output_path):
                                num_entries += 1
                                swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N11_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))

# ### Group-level Results

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N13_UMAP_Eval_Clustering_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N13_UMAP_Eval_Clustering_ALL.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File:  %s' % swarm_path)
print('++ INFO: Logs Folder: %s' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J UMAP_All_SI -f {swarm_path} -b 57 -g 16 -t 4 --time 00:03:00 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
for input_data in input_datas:
    for norm_method in norm_methods:
        for dist in umap_dist_metrics:
            for knn in umap_knns:
                for m in umap_ms:
                    for alpha in umap_alphas:
                        num_iters +=1
                        input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL','UMAP',input_data,'ALL_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.pkl'.format(norm_method=norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   alpha=str(alpha)))
                        output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL','UMAP',input_data,'ALL_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.SI.pkl'.format(norm_method=norm_method, init_method=init_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                        if not osp.exists(output_path):
                            num_entries += 1
                            swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N11_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                            swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))
# -

# *** 
# # Procrustes

import pandas as pd
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm, tqdm_notebook
from utils.basics import norm_methods, input_datas
from utils.io import load_UMAP_SI,load_single_umap

# %%time
si_UMAP = load_UMAP_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])

si_UMAP = si_UMAP.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Alpha','Init','MinDist','Target']).sort_index()


def procrustes_scan_umap_embs(sbj_list,input_data,norm_method,dist,knn,m,alpha, init, mdist,drop_xxxx):
    embs = {}
    # Load all embeddings
    for sbj in sbj_list:
        embs[sbj] = load_single_umap(sbj,input_data,norm_method,dist,knn,alpha,init_method,min_dist,m,drop_xxxx=False)
    # Select best embedding (based on SI) to be used as reference
    si_sel_data = si_UMAP.loc[sbj_list].loc[:,input_data,norm_method,dist,:,m,alpha,init,mdist]
    best        = si_sel_data.sort_values(by='SI', ascending=False).iloc[0].name[0]
    # Get a list with the names of all other scans that need to be transformed
    scans_to_transform = sbj_list.copy()
    scans_to_transform.remove(best)
    # Copy the embedding to use as reference into the ref variable
    ref = embs[best]
    # Create object that will contain all overlapped embeddings
    all_embs            = zscore(ref.copy()).reset_index()
    sub_col = list(np.repeat(best,all_embs.shape[0]))
    # Go one-by-one computing transformation and keeping it 
    for scan in scans_to_transform:
        aux          = embs[scan]
        _, aux_trf,_ = procrustes(ref,aux)
        aux_trf      = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
        all_embs     = all_embs.append(aux_trf).reset_index(drop=True)
        sub_col      = sub_col + list(np.repeat(scan,aux.shape[0]))
    all_embs['Subject'] = sub_col
    # Drop In-between windows if requested
    if drop_xxxx:
        all_embs = all_embs.set_index('Window Name').drop('XXXX').reset_index()
    return all_embs


# %%time
si_procrustes = pd.DataFrame(columns=['Input','Norm','Metric','Knn','m','Alpha','Init','MinDist','Target','SI'])
for input_data in tqdm(input_datas, desc='Input Data:'):
    for norm_method in tqdm(norm_methods, desc='Norm Method', leave=False):
        for m in umap_ms:
            for dist in umap_dist_metrics:
                for knn in umap_knns:
                    for alpha in umap_alphas:
                        aux = procrustes_scan_umap_embs(PNAS2015_subject_list,input_data,norm_method,dist,knn,m,alpha, init_method, min_dist,drop_xxxx=True)
                        aux.index.name = 'WinID'
                        aux.columns.name = ''
                        emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,
                                            'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{md}_a{alpha}_{init}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3),md=min_dist,alpha=str(alpha),
                                                                                                                      wss=str(wss),init=init_method,
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
                        aux.to_pickle(emb_path)
                        umap_dims = [c for c in aux.columns if 'UMAP0' in c]
                        # Compute SI
                        si_sbj  = silhouette_score(aux[umap_dims], aux.reset_index()['Subject'], n_jobs=-1)
                        si_task = silhouette_score(aux[umap_dims], aux.reset_index()['Window Name'], n_jobs=-1)
                        # Write to disk individual SI file
                        df = pd.Series(index=['SI_Subject','SI_Window Name'], dtype=float)
                        df['SI_Subject'] = si_sbj
                        df['SI_Window Name'] = si_task
                        si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{md}_a{alpha}_{init}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                      md=min_dist,alpha=str(alpha),init=init_method,
                                                                                                                      wss=str(wss),
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
                        df.to_pickle(si_path)
                        # Add SI to overall dictionary
                        si_procrustes = si_procrustes.append({'Input':input_data, 'Norm':norm_method, 'Metric':dist,
                                              'Knn':knn, 'm':m, 'Target':'Subject','SI':si_sbj}, ignore_index=True)
                        si_procrustes = si_procrustes.append({'Input':input_data, 'Norm':norm_method, 'Metric':dist,
                                              'Knn':knn, 'm':m, 'Target':'Window Name','SI':si_task}, ignore_index=True)

si_procrustes.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','SI_UMAP_Procrustes.pkl'))


