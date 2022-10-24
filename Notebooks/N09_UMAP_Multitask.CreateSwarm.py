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
# This notebook will compute UMAP for the multi-task dataset. For UMAP we explore three hyper-parameters:
#
# * Distance Function: euclidean, cosine or correlation
# * knn: neighborhood size
# * m: final number of dimensions
# * learning rate: for the optimization phase
#
# Matrices will be written as pandas pickle objects in ```/data/SFIMJGC_HCP7T/manifold_learning/Data_Interim/PNAS2015/{sbj}/UMAP```

import pandas as pd
import numpy as np
import os
import os.path as osp
import getpass
from datetime import datetime
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PRJ_DIR
from utils.basics import umap_dist_metrics, umap_knns, umap_ms, umap_alphas
from utils.basics import input_datas, norm_methods

# + [markdown] tags=[]
# ***
#
# The next cell select the Window Length ```wls``` and Window Step ```wss``` used to generate the matrices
# -

wls      = 45
wss      = 1.5
min_dist = 0.8

# ***
# Those are the norm_methods we will be running

print('++ INFO: Distance Metrics: %s' % str(umap_dist_metrics))
print('++ INFO: Knns:             %s' % str(umap_knns))
print('++ INFO: Ms:               %s' % str(umap_ms))
print('++ INFO: Learning Rates:   %s' % str(umap_alphas))

# The next cell will create the output folders if they do not exist already

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    for input_data in ['Original','Null_ConnRand','Null_PhaseRand']:
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'UMAP',input_data)
        if not osp.exists(path):
            print('++ INFO: Created folder %s' % path)
            os.makedirs(path)

# The next cell will create folders for the swarm log files and for the actual swarm script. Those folders are created using the username as part of their name. That way it is easier for different users to work together on the project.

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Multitask_Scans.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N09_UMAP_Multitask_Scans.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)
# -

# Create swarm script. This script will have one line per matrix to be generated.

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -b 5 -g 4 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        for norm_method in norm_methods:
            for dist in umap_dist_metrics:
                for knn in umap_knns:
                    for m in [2,3]:
                        for alpha in umap_alphas:
                            for init_method in ['spectral']:
                                num_iters +=1
                                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{norm_method}.pkl'.format(subject=subject,norm_method=norm_method, wls=str(int(wls)).zfill(3), wss=str(wss)))
                                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'UMAP',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.pkl'.format(subject=subject,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   norm_method=norm_method,
                                                                                                                                                   alpha=str(alpha)))
                                if not osp.exists(path_out):
                                    num_entries += 1
                                    swarm_file.write('export path_tvfc={path_tvfc} dist={dist} knn={knn} min_dist={min_dist} alpha={alpha} init={init_method} m={m} path_out={path_out}; sh {scripts_dir}/N09_UMAP.sh'.format(path_tvfc=path_tvfc, 
                                                                                                                                    path_out=path_out,
                                                                                                                                    init_method = init_method,
                                                                                                                                    dist=dist,
                                                                                                                                    knn=knn,
                                                                                                                                    m=m, 
                                                                                                                                    min_dist=min_dist,
                                                                                                                                    alpha=alpha,
                                                                                                                                    scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))
# -
# ***
#
# # Group Level Analyses

# Create Output Folders if they do not exists
for input_data in input_datas:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL','UMAP', input_data)
    if not osp.exists(path):
        print('++ INFO: Created folder: %s' % path)
        os.makedirs(path)


# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Multitask_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N09_UMAP_Multitask_ALL.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm path  : %s' % swarm_path)
print('++ INFO: Logs folder : %s' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J UMAP_All_Embs -f {swarm_path} -b 8 -g 16 -t 8 --time 01:00:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
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
                        path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL',input_data,       'ALL_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{norm_method}.pkl'.format(norm_method=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
                        path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL','UMAP',input_data,'ALL_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_spectral.{norm_method}.pkl'.format(norm_method=norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                        if True: #not osp.exists(path_out):
                            num_entries += 1
                            swarm_file.write('export path_tvfc={path_tvfc} dist={dist} knn={knn} min_dist={min_dist} alpha={alpha} m={m} path_out={path_out} init=spectral; sh {scripts_dir}/N09_UMAP.sh'.format(path_tvfc=path_tvfc, 
                                                                                                                                    path_out=path_out,
                                                                                                                                    dist=dist,
                                                                                                                                    knn=knn,
                                                                                                                                    m=m, 
                                                                                                                                    min_dist=min_dist,
                                                                                                                                    alpha=alpha,
                                                                                                                                    scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                            swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))
# -




