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
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PRJ_DIR
from utils.basics import tsne_dist_metrics, tsne_pps, tsne_ms, tsne_alphas, tsne_inits
from utils.basics import input_datas, norm_methods

# + [markdown] tags=[]
# ***
#
# The next cell select the Window Length ```wls``` and Window Step ```wss``` used to generate the matrices
# -

wls      = 45
wss      = 1.5

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
n_jobs=4
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J TSNE_Scan -f {swarm_path} -b 23 -g 4 -t {n_jobs} --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path, n_jobs=n_jobs))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in ['Original']: #input_datas:
    for subject in PNAS2015_subject_list:
        for norm_method in norm_methods:
            for dist in ['correlation']: #tsne_dist_metrics:
                for init_method in tsne_inits:
                    for pp in tsne_pps:
                        for alpha in [10,1000]: #tsne_alphas:
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
# ## 1.2. Compute Silhouette Index on all scan-level TSNE embeddings

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N12_TSNE_Eval_Clustering_Scans.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N12_TSNE_Eval_Clustering_Scans.logs')

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
swarm_file.write('#swarm -J TSNE_Scans_SI -f {swarm_path} -b 424 -g 4 -t 4 --time 00:00:30 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
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
                                if True:# not osp.exists(output_path):
                                    num_entries += 1
                                    swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N11_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -

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
n_jobs=16
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J TSNE_Scan -f {swarm_path} -g 64 -t 16 --partition norm --time 72:00:00 --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path, n_jobs=n_jobs))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in ['Original']:
    for subject in ['ALL']:
        for norm_method in norm_methods:
            for dist in ['correlation']:
                for init_method in tsne_inits:
                    for pp in tsne_pps:
                        for alpha in [10,1000]:
                            for m in [2,3,5,10]:
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
swarm_file.write('#swarm -J TSNE_ALL_SI -f {swarm_path} -b 6 -g 4 -t 4 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in ['Original']:
    for subject in ['ALL']:
        for norm_method in norm_methods:
            for dist in ['correlation']:
                for init_method in tsne_inits:
                    for pp in tsne_pps:
                        for alpha in [10,1000]:
                            for m in [2,3,5,10]:
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
                                if osp.exists(input_path) & (not osp.exists(output_path)):
                                    num_entries += 1
                                    swarm_file.write('export input={input_path} output={output_path}; sh {scripts_dir}/N11_SI.sh'.format(input_path=input_path, 
                                                                                                                     output_path=output_path,
                                                                                                                     scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                                    
                                    swarm_file.write('\n')
swarm_file.close()
print("++ INFO: Missing/Needed = [%d/%d]" % (num_entries,num_iters))
# -

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
swarm_file.write('#swarm -J TSNE_Procrustes_SI -f {swarm_path} -b 6 -g 4 -t 4 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')
num_entries = 0 
num_iters = 0

for input_data in ['Original']:
    for subject in ['Procrustes']:
        for norm_method in norm_methods:
            for dist in ['correlation']:
                for init_method in tsne_inits:
                    for pp in tsne_pps:
                        for alpha in [10,1000]:
                            for m in [2,3,5,10]:
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
                                if True: #(not osp.exists(emb_path)) | (not osp.exists(si_path)):
                                    num_entries += 1
                                    swarm_file.write('export sbj_list="{sbj_list}" input_data={input_data} norm_method={norm_method} dist={dist} pp={pp} m={m} alpha={alpha} init_method={init_method} drop_xxxx={drop_xxxx}; sh {scripts_dir}/N12_TSNE_Procrustes.sh'.format(input_data=input_data,
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
# ***
# # END OF NOTEBOOK
# ***
# ***



# ### Extra cases for the Classification Study

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

_,tsne_best_norm_method,tsne_best_dist, tsne_best_pp, _, tsne_best_alpha,_,_ = si_TSNE.loc[PNAS2015_subject_list,'Original',:,:,:,:,:,:,'Task'].to_xarray().mean(dim='Subject').to_dataframe().sort_values(by='SI',ascending=False).iloc[0].name
print('++ INFO: Best scan-level configuration --> NM=%s, DIST=%s, PP=%d, ALPHA=%f' % (tsne_best_norm_method,tsne_best_dist, tsne_best_pp, tsne_best_alpha))
