# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Embeddings2 + Sdim
#     language: python
#     name: embeddings3
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

# +
wls      = 45
wss      = 1.5
min_dist = 0.8
init_method = 'spectral'

umap_dist_metrics = ['euclidean']
umap_knns         = [70]
umap_ms           = [3]
umap_alphas       = [0.01]
norm_methods      = ['asis']
N_iters           = 1000
# -

# ***
# # 1. Compute UMAP Scan Level Embeddings
#
# ## 1.2. Compute UMAP Embeddings on all input types
# Those are the norm_methods we will be running

print('++ INFO: Distance Metrics: %s' % str(umap_dist_metrics))
print('++ INFO: Knns:             %s' % str(umap_knns))
print('++ INFO: Ms:               %s' % str(umap_ms))
print('++ INFO: Learning Rates:   %s' % str(umap_alphas))

# The next cell will create the output folders if they do not exist already

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    for input_data in ['Original','Null_ConnRand','Null_PhaseRand']:
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'UMAP','Stability')
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

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Multitask_Scans.SWARM.stability.sh')
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
swarm_file.write('#swarm -J UMAP_Scan_Stability -f {swarm_path} -b 20 -g 4 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
for subject in PNAS2015_subject_list:
    for norm_method in norm_methods:
        for dist in umap_dist_metrics:
            for knn in umap_knns:
                for m in umap_ms:
                    for alpha in umap_alphas:
                        for init_method in ['spectral']:
                            for n_iter in range(N_iters):
                                num_iters +=1
                                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{norm_method}.pkl'.format(subject=subject,norm_method=norm_method, wls=str(int(wls)).zfill(3), wss=str(wss)))
                                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'UMAP','Stability','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.I{n_iter}.pkl'.format(subject=subject,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   norm_method=norm_method,
                                                                                                                                                   alpha=str(alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
                                if not osp.exists(path_out):
                                    num_entries += 1
                                    swarm_file.write('export path_tvfc={path_tvfc} dist={dist} knn={knn} min_dist={min_dist} alpha={alpha} init={init_method} m={m} path_out={path_out} stability=True; sh {scripts_dir}/N09_UMAP.sh'.format(path_tvfc=path_tvfc, 
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
# + [markdown] tags=[]
# ## 1.2. Calculate SI on all scan-level UMAP embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Eval_Clustering_Scans.SWARM.stability.sh')
logdir_path    = osp.join(logs_folder, 'N09_UMAP_Eval_Clustering_Scans.logs')

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
swarm_file.write('#swarm -J UMAP_Scans_Stab_SI -f {swarm_path} -b 20 -g 16 -t 4 --time 00:05:00 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
for norm_method in norm_methods:
    for dist in umap_dist_metrics:
        for knn in umap_knns:
            for m in umap_ms:
                for alpha in umap_alphas:
                    for sbj in PNAS2015_subject_list:
                        for n_iter in range(N_iters):
                            num_iters +=1
                            input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP','Stability','{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.I{n_iter}.pkl'.format(norm_method=norm_method,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   alpha=str(alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
                            output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP','Stability','{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.SI.I{n_iter}.pkl'.format(norm_method=norm_method, init_method=init_method,
                                                                                                                                                   sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
                            if osp.exists(input_path) & (not osp.exists(output_path)):
                                num_entries += 1
                                swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N10_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))


