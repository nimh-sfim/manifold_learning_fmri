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
# This notebook will compute TSNE for the multi-task dataset. For UMAP we explore three hyper-parameters:
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
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PRJ_DIR, wls, wss
from utils.basics import tsne_dist_metrics, tsne_pps, tsne_ms, tsne_alphas, tsne_inits
from utils.basics import input_datas, norm_methods

# + [markdown] tags=[]
# ***
# # 1. Compute T-SNE Scan Level Embeddings
#
# ## 1.1. Compute TSNE Embeddings on all input types
# Those are the scenarios we will be running
# -

tsne_dist_metrics = ['correlation']
tsne_pps          = [65]
tsne_ms           = [2]
tsne_alphas       = [10]
norm_methods      = ['asis']
N_iters           = 1000

print('++ INFO: Distance Metrics: %s' % str(tsne_dist_metrics))
print('++ INFO: Perplexitiess:    %s' % str(tsne_pps))
print('++ INFO: Ms:               %s' % str(tsne_ms))
print('++ INFO: Learning Rates:   %s' % str(tsne_alphas))
print('++ INFO: Init Methods:     %s' % str(tsne_inits))

# The next cell will create the output folders if they do not exist already

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE', 'Stability')
        if not osp.exists(path):
            print('++ INFO: Creating folder [%s]' % path)
            os.makedirs(path)

# The next cell will create folders for the swarm log files and for the actual swarm script. Those folders are created using the username as part of their name. That way it is easier for different users to work together on the project.

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N08_TSNE_Multitask_Scans.SWARM.stability.sh')
logdir_path    = osp.join(logs_folder, 'N08_TSNE_Multitask_Scans.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)
# -

# Create swarm script. This script will have one line per matrix to be generated.
#
# > NOTE: For the group level, we will work on extra dimensions (becuase of Procrustes) but only look at Original Data, correlation metric and 10,1000 as learning rate.

# +
# %%time
# Open the file
n_jobs=16
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J TSNE_ScanLevel -f {swarm_path} -b 20 -g 16 -t {n_jobs} --time 00:11:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path, n_jobs=n_jobs))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for subject in PNAS2015_subject_list:
    for norm_method in norm_methods:
        for dist in tsne_dist_metrics:
            for init_method in tsne_inits:
                for pp in tsne_pps:
                    for alpha in tsne_alphas:
                        for m in tsne_ms:
                            for n_iter in range(N_iters):
                                num_iters += 1
                                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original',       '{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
                                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE','Stability','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.I{n_iter}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
                                if not osp.exists(path_out):
                                    num_entries += 1
                                    swarm_file.write('export path_tvfc={path_tvfc} dist={dist} pp={pp} lr={lr} m={m} n_iter=10000 init={init_method} path_out={path_out} n_jobs={n_jobs} norm={norm_method} grad_method=exact stability=True; sh {scripts_dir}/N08_TSNE.sh'.format(path_tvfc=path_tvfc, 
                                                                                                                                    path_out=path_out,
                                                                                                                                    dist=dist,
                                                                                                                                    init_method=init_method,
                                                                                                                                    norm_method=norm_method,
                                                                                                                                    pp=pp,
                                                                                                                                    m=m, 
                                                                                                                                    lr=alpha,
                                                                                                                                    n_jobs=n_jobs,
                                                                                                                                    scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))
# -
# ## 1.2. Compute Silhouette Index on all scan-level TSNE embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N08_TSNE_Eval_Clustering_Scans.SWARM.stability.sh')
logdir_path    = osp.join(logs_folder, 'N08_TSNE_Eval_Clustering_Scans.logs')

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
swarm_file.write('#swarm -J TSNE_Scans_SI_Orig -f {swarm_path} -b 20 -g 4 -t 4 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for subject in PNAS2015_subject_list:
    for norm_method in norm_methods:
        for dist in tsne_dist_metrics:
            for init_method in tsne_inits:
                for pp in tsne_pps:
                    for alpha in tsne_alphas:
                        for m in tsne_ms:
                            for n_iter in range(N_iters):                              
                                num_iters += 1
                                input_path  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE','Stability','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.I{n_iter}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
                                output_path  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE','Stability','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.SI.I{n_iter}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha),
                                                                                                                                                   n_iter=str(n_iter).zfill(5)))
                                if not osp.exists(output_path):
                                    num_entries += 1
                                    swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N10_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -


