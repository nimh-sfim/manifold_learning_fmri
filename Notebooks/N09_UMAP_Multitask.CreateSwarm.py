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
init_method = 'spectral'

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
swarm_file.write('#swarm -J UMAP_Scan -f {swarm_path} -b 5 -g 4 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        for norm_method in norm_methods:
            for dist in umap_dist_metrics:
                for knn in umap_knns:
                    for m in umap_ms:
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
                                if True: #not osp.exists(path_out):
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
# ## 1.2. Calculate SI on all scan-level UMAP embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Eval_Clustering_Scans.SWARM.sh')
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
swarm_file.write('#swarm -J UMAP_Scans_SI -f {swarm_path} -b 346 -g 16 -t 4 --time 00:00:40 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
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
                            if osp.exists(input_path) & (not osp.exists(output_path)):
                                num_entries += 1
                                swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N10_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))

# ***
#
# # 2. UMAP Group-Level Embeddings
#
# ## 2.1 Compute Group Embeddings via "Concatenation + UMAP"

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Clustering_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N09_UMAP_Clustering_ALL.logs')

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
swarm_file.write('#swarm -J UMAP_ALL -f {swarm_path} -b 5 -g 64 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
for subject in ['ALL']:
    for input_data in input_datas:
        for norm_method in norm_methods:
            for dist in umap_dist_metrics:
                for knn in umap_knns:
                    for m in umap_ms:
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
                                if True:#not osp.exists(path_out):
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

# ## 2.2. Compute SI for UMAP Group results: "Concatenation + UMAP"

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Eval_Clustering_ALL.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N09_UMAP_Eval_Clustering_ALL.logs')

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
swarm_file.write('#swarm -J UMAP_All_SI -f {swarm_path} -b 18 -g 16 -t 4 --time 00:13:20 --partition=quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
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
                            swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N10_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                            swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Attempts/Written = [%d/%d]" % (num_entries,num_iters))
# -

# ***
# # 3. Group Level: UMAP + Procrustes (Generation + SI in the same program)

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N09_UMAP_Clustering_Procrustes.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N09_UMAP_Clustering_Procrustes.logs')

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
swarm_file.write('#swarm -J UMAP_Procrustes_SI -f {swarm_path} -b 6 -g 4 -t 4 --time 00:40:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

input_method = 'spectral'
mdist        = 0.8
for input_data in ['Original']:
    for norm_method in norm_methods:
        for dist in umap_dist_metrics:
            for knn in umap_knns:
                for alpha in umap_alphas:
                    for m in umap_ms:
                        num_iters += 1
                        emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,
                                'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{mdist}_a{alpha}_{init}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3),alpha=str(alpha),
                                                                                                                      wss=str(wss),init=init_method,
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      mdist=str(mdist),
                                                                                                                      m=str(m).zfill(4)))
                        si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,
                               'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{mdist}_a{alpha}_{init}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                      alpha=str(alpha),init=init_method,
                                                                                                                      wss=str(wss),
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      mdist=str(mdist),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))

                        if (not osp.exists(emb_path)) | (not osp.exists(si_path)):
                            num_entries += 1
                            swarm_file.write('export sbj_list="{sbj_list}" input_data={input_data} norm_method={norm_method} dist={dist} knn={knn} mdist={mdist} m={m} drop_xxxx={drop_xxxx} alpha={alpha} init_method={init_method}; sh {scripts_dir}/N11_UMAP_Procrustes.sh'.format(
                                                                                                 sbj_list=','.join(PNAS2015_subject_list),
                                                                                                 input_data = input_data,
                                                                                                 norm_method = norm_method,
                                                                                                 dist = dist,
                                                                                                 knn  = str(knn),
                                                                                                 m = str(m),
                                                                                                 mdist = str(mdist),
                                                                                                 init_method=init_method,
                                                                                                 alpha=str(alpha),
                                                                                                 drop_xxxx='False',
                                                                                                 scripts_dir = osp.join(PRJ_DIR,'Notebooks')))
                            swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -

# ***
# # 4. Save all Computed SI values into a single dataframe

from utils.io import load_UMAP_SI

# %%time
RELOAD_SI_UMAP = True
if RELOAD_SI_UMAP:
    print('++ Loading Group-Level "Concat + UMAP" SI values.....')
    si_UMAP_all = load_UMAP_SI(sbj_list=['ALL'],check_availability=False, verbose=True, wls=wls, wss=wss, input_datas=['Original'])
    print('++ ==================================================')
    print('++ Loading Group-Level "UMAP + Procrustes" SI values.....')
    si_UMAP_procrustes = load_UMAP_SI(sbj_list=['Procrustes'],check_availability=False, verbose=True, wls=wls, wss=wss, input_datas=['Original'])
    print('++ ======================================================')
    print('++ Loading Scan-Level SI values.....')
    si_UMAP_scans = load_UMAP_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=True, wls=wls, wss=wss)
    print('++ =================================')
    print('++ Combine into a single DataFrame....')
    si_UMAP = pd.concat([si_UMAP_scans, si_UMAP_all, si_UMAP_procrustes])
    si_UMAP.replace('Window Name','Task', inplace=True)
    si_UMAP = si_UMAP.set_index(['Subject','Input Data','Norm','Init','MinDist','Metric','Knn','Alpha','m','Target']).sort_index()
    del si_UMAP_scans, si_UMAP_all, si_UMAP_procrustes
    si_path = osp.join(PRJ_DIR,'Dashboard','Data','si_UMAP.pkl')
    print('++ Save Dataframe to disk [%s]' % si_path)
    si_UMAP.to_pickle(si_path)
else:
    si_UMAP = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_UMAP.pkl'))

# ***
# ***
# # END OF NOTEBOOK
# ***
# ***



# ### Extra cases for the Classification Study

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J UMAP_Scan_Extra -f {swarm_path} -g 4 -t 4 --time 04:00:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        for norm_method in ['asis']:
            for dist in ['euclidean']:
                for knn in [70]:
                    for m in [5,10,15,20,25]:
                        for alpha in [0.01]:
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
