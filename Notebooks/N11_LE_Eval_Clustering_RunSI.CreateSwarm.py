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
from utils.basics import le_dist_metrics, le_knns, le_ms
from utils.basics import PRJ_DIR, PNAS2015_subject_list
from datetime import datetime

import os.path as osp
import os
import getpass
# -

wls = 45
wss = 1.5

print('++ INFO: Distance Metrics: %s' % str(le_dist_metrics))
print('++ INFO: Knns:             %s' % str(le_knns))
print('++ INFO: Ms:               %s' % str(le_ms))

# ***
# # 1. Scan-level Results

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N11_LE_Eval_Clustering_Scans.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N11_LE_Eval_Clustering_Scans.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print("++ INFO: Swarm File : %s" % swarm_path)
print("++ INFO: Logs Folder: %s" % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -b 44 -g 16 -t 4 --time 00:05:00 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
for scenario in ['asis','zscored']:
    for dist in le_dist_metrics:
        for knn in le_knns:
            for m in [5,10,15,20,25]:
                for sbj in PNAS2015_subject_list:
                    for input_data in ['Null_ConnRand','Null_PhaseRand']: #['Original','Null_ConnRand','Null_PhaseRand']:
                        input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(scenario=scenario,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                        output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.SI.pkl'.format(scenario=scenario,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                        swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N11_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                        swarm_file.write('\n')
swarm_file.close()
# -

# ***
# # 2. Group-Level Results

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N11_LE_Eval_Clustering_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N11_LE_Eval_Clustering_ALL.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print("++ INFO: Swarm File : %s" % swarm_path)
print("++ INFO: Logs Folder: %s" % logdir_path)
# -

# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
needed_files, missing_files = 0,0
# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -b 57 -g 16 -t 4 --time 00:03:00 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
for scenario in ['asis','zscored']:
    for dist in le_dist_metrics:
        for knn in le_knns:
            for m in le_ms:
                for input_data in ['Original','Null_ConnRand','Null_PhaseRand']:
                    needed_files += 1
                    input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL','LE',input_data,'ALL_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(scenario=scenario,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                    output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL','LE',input_data,'ALL_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.SI.pkl'.format(scenario=scenario,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                    if True: #osp.exists(input_path) & (not osp.exists(output_path)):
                        missing_files += 1
                        swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N11_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                        swarm_file.write('\n')
swarm_file.close()
print(missing_files,needed_files)

# *** 
# # Procrustes

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N11_LE_Eval_Clustering_Procrustes.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N11_LE_Eval_Clustering_Procrustes.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print("++ INFO: Swarm File : %s" % swarm_path)
print("++ INFO: Logs Folder: %s" % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J LE_Procrustes -f {swarm_path} -b 57 -g 16 -t 4 --time 00:03:00 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
for norm_method in ['asis','zscored']:
    for dist in le_dist_metrics:
        for knn in le_knns:
            for m in le_ms + [5,10,15,20,25]:
                for input_data in ['Original','Null_ConnRand','Null_PhaseRand']:
                    swarm_file.write('export sbj_list="{sbj_list}" input_data={input_data} norm_method={norm_method} dist={dist} knn={knn} m={m} drop_xxxx={drop_xxxx}; sh {scripts_dir}/N11_LE_Procrustes.sh'.format(
                                                                                                 sbj_list=','.join(PNAS2015_subject_list),
                                                                                                 input_data = input_data,
                                                                                                 norm_method = norm_method,
                                                                                                 dist = dist,
                                                                                                 knn  = str(knn),
                                                                                                 m = str(m),
                                                                                                 drop_xxxx='False',
                                                                                                 scripts_dir = osp.join(PRJ_DIR,'Notebooks')))
                    swarm_file.write('\n')
swarm_file.close()
# -
import numpy as np
np.arange(2,4)

np.arange(2,3)














import pandas as pd
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm, tqdm_notebook
from utils.basics import norm_methods, input_datas
from utils.io import load_single_le, load_LE_SI



















# %%time
si_LE = load_LE_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3,5,10,15,20,25,30],input_datas=['Original'])

si_LE = si_LE.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()

si_LE['Subject']


def procrustes_scan_embs(sbj_list,input_data,scenario,dist,knn,m,drop_xxxx):
    embs = {}
    # Load all embeddings
    for sbj in sbj_list:
        embs[sbj] = load_single_le(sbj,input_data,scenario,dist,knn,m, drop_xxxx=False)
    # Select best embedding (based on SI) to be used as reference
    si_sel_data = si_LE.loc[sbj_list].loc[:,input_data,scenario,dist,:,m]
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
si_procrustes = pd.DataFrame(columns=['Input','Norm','Metric','Knn','m','Target','SI'])
for input_data in tqdm(input_datas, desc='Input Data:'):
    for norm_method in tqdm(norm_methods, desc='Norm Method', leave=False):
        for m in [5]:
            for dist in le_dist_metrics:
                for knn in le_knns:
                    aux = procrustes_scan_embs(PNAS2015_subject_list,input_data,norm_method,dist,knn,m,drop_xxxx=True)
                    aux.index.name = 'WinID'
                    aux.columns.name = ''
                    emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','LE',input_data,
                                        'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                      wss=str(wss),
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
                    aux.to_pickle(emb_path)
                    le_dims = [c for c in aux.columns if 'LE0' in c]
                    # Compute SI
                    si_sbj  = silhouette_score(aux[le_dims], aux.reset_index()['Subject'], n_jobs=-1)
                    si_task = silhouette_score(aux[le_dims], aux.reset_index()['Window Name'], n_jobs=-1)
                    # Write to disk individual SI file
                    df = pd.Series(index=['SI_Subject','SI_Window Name'], dtype=float)
                    df['SI_Subject'] = si_sbj
                    df['SI_Window Name'] = si_task
                    si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','LE',input_data,
                                        'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
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




si_procrustes.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','SI_LE_Procrustes.pkl'))

si_procrustes_indexes = si_procrustes.copy().set_index(['Input','Norm','Metric','Knn','m','Target']).sort_index()

si_procrustes_indexes.head(5)

import hvplot.pandas

si_procrustes_indexes.hvplot.kde('SI', by='Input') 

si_procrustes_indexes.loc['Original']
