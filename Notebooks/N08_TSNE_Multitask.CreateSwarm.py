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

# ***
# # 1. Compute T-SNE Scan Level Embeddings
#
# ## 1.1. Compute TSNE Embeddings on all input types
# Those are the scenarios we will be running

print('++ INFO: Distance Metrics: %s' % str(tsne_dist_metrics))
print('++ INFO: Perplexitiess:    %s' % str(tsne_pps))
print('++ INFO: Ms:               %s' % str(tsne_ms))
print('++ INFO: Learning Rates:   %s' % str(tsne_alphas))
print('++ INFO: Init Methods:     %s' % str(tsne_inits))

# The next cell will create the output folders if they do not exist already

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE', input_data)
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

swarm_path     = osp.join(swarm_folder,'N08_TSNE_Multitask_Scans.SWARM.sh')
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
swarm_file.write('#swarm -J TSNE_ScanLevel -f {swarm_path} -b 21 -g 16 -t {n_jobs} --time 00:11:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path, n_jobs=n_jobs))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in input_datas:
    for subject in PNAS2015_subject_list:
        for norm_method in norm_methods:
            for dist in tsne_dist_metrics:
                for init_method in tsne_inits:
                    for pp in tsne_pps:
                        for alpha in tsne_alphas:
                            for m in tsne_ms:
                                num_iters += 1
                                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,input_data,       '{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
                                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                if not osp.exists(path_out):
                                    num_entries += 1
                                    swarm_file.write('export path_tvfc={path_tvfc} dist={dist} pp={pp} lr={lr} m={m} n_iter=10000 init={init_method} path_out={path_out} n_jobs={n_jobs} norm={norm_method} grad_method=exact; sh {scripts_dir}/N08_TSNE.sh'.format(path_tvfc=path_tvfc, 
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
# > **NOTE**: m=2 all completed (12/06/2022) | m=3 all completed (12/06/2022) | m=5 all completed (12/06/2022) | m=10 all completed (12/06/2022). This is for the whole hyper-parameter space.

# ## 1.2. Compute Silhouette Index on all scan-level TSNE embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N08_TSNE_Eval_Clustering_Scans.SWARM.sh')
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
swarm_file.write('#swarm -J TSNE_Scans_SI_Orig -f {swarm_path} -b 21 -g 4 -t 4 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in input_datas:
    for subject in PNAS2015_subject_list:
        for norm_method in norm_methods:
            for dist in tsne_dist_metrics:
                for init_method in tsne_inits:
                    for pp in tsne_pps:
                        for alpha in tsne_alphas:
                            for m in tsne_ms:                                
                                num_iters += 1
                                input_path  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                output_path  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.SI.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                if not osp.exists(output_path):
                                    num_entries += 1
                                    swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N10_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -

# > **NOTE:** m=2 (12/06/2022) | m=3 (12/06/2022) | m=5 (12/06/2022) | m=10 (12/06/2022)

# ***
#
# # 2. Group-level TSNE: "Concatenation + TSNE"
#
# ## 2.1 Create Group-level "Concatenation + TSNE" Embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N08_TSNE_Multitask_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N08_TSNE_Multitask_ALL.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)

# +
# %%time
# Open the file
n_jobs=24
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J TSNE_15_ALL -f {swarm_path} -g 128 -t 64 --partition norm --time 7-00:00:00 --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path, n_jobs=n_jobs))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in ['Original']:
    for subject in ['ALL']:
        for norm_method in norm_methods:
            for dist in ['correlation']:
                for init_method in tsne_inits:
                    for pp in [5,10,15]:
                        for alpha in [10,1000]:
                            for m in [15]:
                                num_iters += 1
                                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,input_data,       '{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
                                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                if not osp.exists(path_out):
                                    num_entries += 1
                                    swarm_file.write('export path_tvfc={path_tvfc} dist={dist} pp={pp} lr={lr} m={m} n_iter=10000 init={init_method} path_out={path_out} n_jobs={n_jobs} norm={norm_method} grad_method=exact; sh {scripts_dir}/N08_TSNE.sh'.format(path_tvfc=path_tvfc, 
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

# > **NOTE:** Given computational needs, we only do this part for the following hyper-parameter set: Original data, correlation distance, lr = 10 or 1000, m =2,3,5,10, all norms and all pps
#
# > **NOTE:** m=2 (12/06/2022) | m=3 (12/06/2022) | m=5 (12/06/2022) | m=10 (12/06/2022 - 3 cases missing)
#
# > **NOTE:** Need to update results with PP=5,10 and 15 for the concat (12/08/2022)

# ## 2.2. Evaluate Group-level "Concatenation + TSNE" Embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N12_TSNE_Eval_Clustering_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N12_TSNE_Eval_Clustering_ALL.logs')

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
swarm_file.write('#swarm -J TSNE_ALL_5_SI -f {swarm_path} -b 6 -g 4 -t 4 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for subject in ['ALL']:
    for input_data in ['Original']:
        for norm_method in norm_methods:
            for dist in ['correlation']:
                for init_method in tsne_inits:
                    for pp in [5,10,15]:
                        for alpha in [10,1000]:
                            for m in [15]:
                                num_iters += 1
                                input_path  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                output_path  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'TSNE',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.SI.pkl'.format(subject=subject,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                #if osp.exists(input_path) & (not osp.exists(output_path)):
                                if (not osp.exists(output_path)):
                                    num_entries += 1
                                    swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N10_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -

# > **NOTE:** m=2 (12/06/2022) | m=3 (12/06/2022) | m=5 (12/06/2022) | m=10 (12/06/2022) --> Missing the same 3 that need to complete

# # 2.3 Create and Evalaute group-level TSNE: "TSNE + Procrustes"

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N12_TSNE_Eval_Clustering_Procrustes.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N12_TSNE_Eval_Clustering_Procrustes.logs')

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
swarm_file.write('#swarm -J TSNE_Procrustes_SI -f {swarm_path} -b 14 -g 4 -t 4 --time 00:17:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in ['Original']:
    for subject in ['Procrustes']:
        for norm_method in norm_methods:
            for dist in ['correlation']:
                for init_method in tsne_inits:
                    for pp in [5,10,15]:
                        for alpha in [10,1000]:
                            for m in [2,3,5,10,15,20,25,30]:
                                num_iters += 1
                                emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','TSNE',input_data,
                                                            'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3),alpha=str(alpha),
                                                                                                                      wss=str(wss),init=init_method,
                                                                                                                      dist=dist,
                                                                                                                      pp=str(pp).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
                                si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','TSNE',input_data,
                                                           'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                      alpha=str(alpha),init=init_method,
                                                                                                                      wss=str(wss),
                                                                                                                      dist=dist,
                                                                                                                      pp=str(pp).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
                                if (not osp.exists(emb_path)) | (not osp.exists(si_path)):
                                    num_entries += 1
                                    swarm_file.write('export sbj_list="{sbj_list}" input_data={input_data} norm_method={norm_method} dist={dist} pp={pp} m={m} alpha={alpha} init_method={init_method} drop_xxxx={drop_xxxx}; sh {scripts_dir}/N11_TSNE_Procrustes.sh'.format(input_data=input_data,
                                                                                                                     sbj_list=','.join(PNAS2015_subject_list),
                                                                                                                     norm_method=norm_method,
                                                                                                                     dist=dist,
                                                                                                                     pp=str(pp),
                                                                                                                     m=str(m),
                                                                                                                     alpha=str(alpha),
                                                                                                                     init_method = init_method,
                                                                                                                     drop_xxxx = 'False',
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -
# ***
# # Load and save into a single dataframe

from utils.io import load_TSNE_SI

si_TSNE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))
print(si_TSNE.shape)

# %%time
print('++ INFO: Loading SI for Concat + TSNE....')
si_TSNE_all              = load_TSNE_SI(sbj_list=['ALL'],               check_availability=False, verbose=False, wls=wls, wss=wss, ms=[2,3,5,10], dist_metrics=['correlation'], input_datas=['Original'], alphas=[10,1000])

print('++ INFO: Loading SI for TSNE + Procrustes....')
si_TSNE_procrustes       = load_TSNE_SI(sbj_list=['Procrustes'],        check_availability=False, verbose=True, wls=wls, wss=wss, ms=[2,3,5,10,15,20,25,30], dist_metrics=['correlation'], input_datas=['Original'], alphas=[10,1000])

# %%time
print('++ INFO: Loading SI for scan level TSNE...')
si_TSNE_scans            = load_TSNE_SI(sbj_list=PNAS2015_subject_list, check_availability=False, verbose=True, wls=wls, wss=wss, ms=[2,3,5,10])

si_TSNE = pd.concat([si_TSNE_scans, si_TSNE_all, si_TSNE_procrustes])
si_TSNE.replace('Window Name','Task', inplace=True)
si_TSNE = si_TSNE.set_index(['Subject','Input Data','Norm','Metric','PP','m','Alpha','Init','Target']).sort_index()
si_TSNE.to_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))

# ***
# ***
# # END OF NOTEBOOK
# ***
# ***
