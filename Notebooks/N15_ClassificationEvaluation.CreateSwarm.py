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

# # Description
#
# This notebook computes classification accuracy on a couple of scenarios to demonstrate the value of keeping dimensions above 3.

import pandas as pd
import numpy as np
import os
import os.path as osp
import getpass
from datetime import datetime
from tqdm.notebook import tqdm
from utils.basics import task_cmap_caps
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PNAS2015_roi_names_path, PNAS2015_win_names_paths, PRJ_DIR, input_datas, norm_methods
from utils.basics import umap_ms, umap_knns, le_knns,le_ms
import xarray as xr
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from utils.basics import wls, wss, tr

# + [markdown] tags=[]
# # 1. UMAP
# -

umap_min_dist    = 0.8
umap_init_method = 'spectral'
tsne_init_method = 'pca'

# After looking at the clustering evaluation results, we will select: Euclidean, knn > 50 and alpha = 0.01

umap_cl_dist, umap_cl_alpha, umap_cl_mdist = 'euclidean',0.01, 0.8
umap_cl_knns                = [knn for knn in umap_knns if knn > 50]

# Create output folder

for input_data in ['Original']:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','UMAP',input_data)
    if not osp.exists(path):
        print('++ INFO: Creating new folder [%s]' % path)
        os.makedirs(path)

# Create folders and files for batch jobs

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N16_Figure10_ClassificationEval_UMAP.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N16_Figure10_ClassificationEval_UMAP.logs')

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
swarm_file.write('#swarm -J Clf_Group_UMAP_Procrustes -f {swarm_path} -b 4 -g 8 -t 8 --time 01:00:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
n_total, n_needed = 0,0
input_data = 'Original'
for clf in 'logisticregression','svc':
    for m in [2,3,5,10,15,20,25,30]:
        for knn in umap_cl_knns:
            for nm in norm_methods:
                n_total += 1
                features = ','.join(['UMAP'+str(i+1).zfill(3) for i in range(m)])
                input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,
                                  'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{md}_a{alpha}_spectral.{nm}.pkl'.format(nm=nm,dist=umap_cl_dist,knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),md=str(umap_cl_mdist),
                                                                                                                                                   alpha=str(umap_cl_alpha),
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss)))
                output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','UMAP',input_data,
                                   'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}.{nm}.clf_results.{clf}_WindowName.pkl'.format(nm=nm,dist=umap_cl_dist,knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),md=str(umap_cl_mdist),clf=clf,
                                                                                                                                                   alpha=str(umap_cl_alpha),
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss)))
                if osp.exists(input_path) & (not osp.exists(output_path)):
                    n_needed += 1
                    swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                       input_path = input_path, output_path=output_path, clf=clf, features=features,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                    swarm_file.write('\n')
swarm_file.close()
print('[%d/%d]' % (n_needed,n_total))
# -

# # 2. Laplacian Eigenmaps
# After looking at the clustering evaluation results, we will select: Euclidean, knn > 50.

le_cl_dist = 'correlation'
le_cl_knns                = [knn for knn in le_knns if knn > 50]

for input_data in ['Original']:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','LE',input_data)
    if not osp.exists(path):
        print('++ INFO: Creating new folder [%s]' % path)
        os.makedirs(path)

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N15_Classify_GroupLevel_Procrustes_LE.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N15_Classify_GroupLevel_Procrustes_LE.logs')

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
swarm_file.write('#swarm -J Clf_Group_LE_Procrustes -f {swarm_path} -b 4 -g 8 -t 8 --time 01:00:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
n_total, n_needed = 0,0
input_data='Original'
for clf in 'logisticregression','svc':
    for m in [2,3,5,10,15,20,25,30]:
        for knn in le_cl_knns:
            for nm in norm_methods:
                n_total += 1
                features = ','.join(['LE'+str(i+1).zfill(3) for i in range(m)])
                input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','LE',input_data,
                                      'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.pkl'.format(nm=nm, dist=le_cl_dist,knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss)))
                output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','LE',input_data,
                                       'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.clf_results.{clf}_WindowName.pkl'.format(nm=nm,dist=le_cl_dist,knn=str(knn).zfill(4),
                                                                                                                                                   clf=clf,
                                                                                                                                                   wls=str(int(wls)).zfill(3),m=str(m).zfill(4),
                                                                                                                                                   wss=str(wss)))
                if osp.exists(input_path) & (not osp.exists(output_path)):
                    n_needed += 1
                    swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                       input_path = input_path, output_path=output_path, clf=clf, features=features,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                    swarm_file.write('\n')
swarm_file.close()
print('[%d/%d]' % (n_needed,n_total))
