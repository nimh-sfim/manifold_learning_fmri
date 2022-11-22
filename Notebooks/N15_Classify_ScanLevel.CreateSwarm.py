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

import pandas as pd
import numpy as np
import os
import os.path as osp
import getpass
from datetime import datetime
from tqdm.notebook import tqdm
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PNAS2015_roi_names_path, PNAS2015_win_names_paths, PRJ_DIR, input_datas, norm_methods

wls = 45
wss = 1.5
tr  = 1.5
win_names_path = PNAS2015_win_names_paths[(wls,wss)]
umap_min_dist    = 0.8
umap_init_method = 'spectral'
tsne_init_method = 'pca'

# Given that we will use SI as a way to check optimal embeddings, we first load the results for the three modalities

# %%time
RELOAD_SI_LE = False
if RELOAD_SI_LE:
    si_LE_all        = load_LE_SI(sbj_list=['ALL'],check_availability=False, verbose=True, wls=wls, wss=wss, ms=[2,3])
    si_LE_procrustes = load_LE_SI(sbj_list=['Procrustes'],check_availability=False, verbose=True, wls=wls, wss=wss, ms=[2,3])
    si_LE_scans      = load_LE_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_LE            = pd.concat([si_LE_scans, si_LE_all, si_LE_procrustes])
    si_LE.replace('Window Name','Task', inplace=True)
    si_LE            = si_LE.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()
    si_LE.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_LE.pkl'))
    del si_LE_scans, si_LE_all, si_LE_procrustes
else:
    si_LE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_LE.pkl'))

# + tags=[]
# %%time
RELOAD_SI_UMAP = False
if RELOAD_SI_UMAP:
    si_UMAP_all = load_UMAP_SI(sbj_list=['ALL'],check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_UMAP_procrustes = load_UMAP_SI(sbj_list=['Procrustes'],check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_UMAP_scans = load_UMAP_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=True, wls=wls, wss=wss, ms=[2,3])
    
    si_UMAP = pd.concat([si_UMAP_scans, si_UMAP_all, si_UMAP_procrustes])
    si_UMAP.replace('Window Name','Task', inplace=True)
    si_UMAP = si_UMAP.set_index(['Subject','Input Data','Norm','Init','MinDist','Metric','Knn','Alpha','m','Target']).sort_index()
    del si_UMAP_scans, si_UMAP_all, si_UMAP_procrustes
    
    si_UMAP.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_UMAP.pkl'))
else:
    si_UMAP = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_UMAP.pkl'))
# -

# %%time
RELOAD_SI_TSNE = False
if RELOAD_SI_TSNE:
    si_TSNE_all        = load_TSNE_SI(sbj_list=['ALL'],               check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_TSNE_procrustes = load_TSNE_SI(sbj_list=['Procrustes'],        check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    si_TSNE_scans      = load_TSNE_SI(sbj_list=PNAS2015_subject_list, check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3])
    
    si_TSNE = pd.concat([si_TSNE_scans, si_TSNE_all, si_TSNE_procrustes])
    si_TSNE.replace('Window Name','Task', inplace=True)
    si_TSNE = si_TSNE.set_index(['Subject','Input Data','Norm','Metric','PP','m','Alpha','Init','Target']).sort_index()
    del si_TSNE_scans, si_TSNE_all, si_TSNE_procrustes
    
    si_TSNE.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))
else:
    si_TSNE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))

# ***
# # Scan-Level

# + tags=[]
# Create Output Folders if they do not exists
for sbj in PNAS2015_subject_list:
    for emb_tech in ['LE','TSNE','UMAP','SWC']:
        for input_data in input_datas:
            path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Classification',emb_tech,input_data)
            if not osp.exists(path):
                print('++ INFO: Created folder %s' % path)
                os.makedirs(path)
# -

# ## Scan-Level - Laplacian EigenMaps

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N15_Classify_ScanLevel_LE.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N15_Classify_ScanLevel_LE.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)
# -

input_data = 'Original' 
_,best_nm, best_dist, best_knn, _, _ = si_LE.loc[PNAS2015_subject_list,input_data,:,:,:,:,'Task'].to_xarray().mean(dim='Subject').to_dataframe().sort_values(by='SI',ascending=False).iloc[0].name
print('++ INFO: Scenario selected for classification: [nm=%s, dist=%s, knn=%d]' % (best_nm, best_dist, best_knn))

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J Classify_Scan -f {swarm_path} -b 20 -g 8 -t 8 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for sbj in PNAS2015_subject_list:
    for input_data in input_datas:
        for clf in 'logisticregression','svc':
            for m in [2,3,5,10,15,20,25]:
                features = ','.join(['LE'+str(i+1).zfill(3) for i in range(m)])
                input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m0030.{nm}.pkl'.format(nm=best_nm,dist=best_dist,knn=str(best_knn).zfill(4),
                                                                                                                                                   sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss)))
                output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Classification','LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.clf_results.{clf}_WindowName.pkl'.format(nm=best_nm,dist=best_dist,knn=str(best_knn).zfill(4),
                                                                                                                                                   sbj=sbj,clf=clf,
                                                                                                                                                   wls=str(int(wls)).zfill(3),m=str(m).zfill(4),
                                                                                                                                                   wss=str(wss)))
                                        
                swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                       input_path = input_path, output_path=output_path, clf=clf, features=features,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()
# -

# ## Scan-Level - UMAP

# +
#min_dist = 0.8
#init_method = 'spectral'
# -

_, umap_best_norm_method, _, _, umap_best_dist, umap_best_knn, umap_best_alpha, _ , _ = si_UMAP.loc[PNAS2015_subject_list,'Original',:,:,:,:,:,:,:,'Task'].to_xarray().mean(dim='Subject').to_dataframe().sort_values(by='SI',ascending=False).iloc[0].name
print('++ INFO: Scenario selected for classification: [nm=%s, dist=%s, knn=%d, alpha=%f]' % (umap_best_norm_method, umap_best_dist, umap_best_knn,umap_best_alpha ))

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N15_Classify_ScanLevel_UMAP.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N15_Classify_ScanLevel_UMAP.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J Classify_Scan_UMAP -f {swarm_path} -b 20 -g 8 -t 8 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for sbj in PNAS2015_subject_list:
    for input_data in input_datas:
        for clf in 'logisticregression','svc':
            for m in [2,3,5,10,15,20,25]:
                features = ','.join(['UMAP'+str(i+1).zfill(3) for i in range(m)])
                input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{nm}.pkl'.format(sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=umap_best_dist,
                                                                                                                                                   knn=str(umap_best_knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   nm=umap_best_norm_method,
                                                                                                                                                   alpha=str(umap_best_alpha)))
                output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Classification','UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{nm}.clf_results.{clf}_WindowName.pkl'.format(sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), clf=clf,
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=umap_best_dist,
                                                                                                                                                   knn=str(umap_best_knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   nm=umap_best_norm_method,
                                                                                                                                                   alpha=str(umap_best_alpha)))                                        
                swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                       input_path = input_path, output_path=output_path, clf=clf, features=features,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()
# -

# # Scan Level - TSNE

# +
#init_method = 'pca'
# -

_,tsne_best_norm_method,tsne_best_dist, tsne_best_pp, _, tsne_best_alpha,_,_ = si_TSNE.loc[PNAS2015_subject_list,'Original',:,:,:,:,:,:,'Task'].to_xarray().mean(dim='Subject').to_dataframe().sort_values(by='SI',ascending=False).iloc[0].name
print('++ INFO: Best scan-level configuration --> NM=%s, DIST=%s, PP=%d, ALPHA=%f' % (tsne_best_norm_method,tsne_best_dist, tsne_best_pp, tsne_best_alpha))

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N15_Classify_ScanLevel_TSNE.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N15_Classify_ScanLevel_TSNE.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J Classify_Scan_TSNE -f {swarm_path} -b 20 -g 8 -t 8 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for sbj in PNAS2015_subject_list:
    for input_data in input_datas:
        for clf in 'logisticregression','svc':
            for m in [2,3,5,10,15,20,25]:
                features   = ','.join(['TSNE'+str(i+1).zfill(3) for i in range(m)])
                input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'TSNE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.pkl'.format(sbj=sbj,
                                                                                                                                                   nm = tsne_best_norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=tsne_best_dist,
                                                                                                                                                   pp=str(tsne_best_pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(tsne_best_alpha)))
                output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Classification','TSNE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.clf_results.{clf}_WindowName.pkl'.format(sbj=sbj,
                                                                                                                                                   nm = tsne_best_norm_method,clf=clf,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=tsne_best_dist,
                                                                                                                                                   pp=str(tsne_best_pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(tsne_best_alpha)))                                 
                swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                       input_path = input_path, output_path=output_path, clf=clf, features=features,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()
# -

# ***
# # Scan Level - SWC Directly

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N15_Classify_ScanLevel_SWC.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N15_Classify_ScanLevel_SWC.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J Classify_Scan_SWC -f {swarm_path} -b 5 -g 8 -t 8 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for sbj in PNAS2015_subject_list:
    for input_data in input_datas:
        for norm_method in ['asis','zscored']:
            for mat_type in ['Z','R']:
                for clf in 'logisticregression','svc':
                        features   = 'All_Connections'
                        input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.{mat_type}.{nm}.pkl'.format(sbj=sbj,
                                                                                                                                                               nm = norm_method,
                                                                                                                                                               wls=str(int(wls)).zfill(3), 
                                                                                                                                                               wss=str(wss),
                                                                                                                                                               mat_type=mat_type))
                        output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'Classification','SWC',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.{mat_type}.{nm}.clf_results.{clf}_WindowName.pkl'.format(sbj=sbj,
                                                                                                                                                               nm = norm_method,
                                                                                                                                                               wls=str(int(wls)).zfill(3), 
                                                                                                                                                               wss=str(wss),
                                                                                                                                                               mat_type=mat_type,
                                                                                                                                                               clf=clf))                   
                        swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                               input_path = input_path, output_path=output_path, clf=clf, features=features,
                               scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                        swarm_file.write('\n')
swarm_file.close()
