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
# This notebook will compute LE for the multi-task dataset. For LE we explore three hyper-parameters:
#
# * Distance Function: euclidean, cosine or correlation
# * knn: neighborhood size
# * m: final number of dimensions
#
# Matrices will be written as pandas pickle objects in ```/data/SFIMJGC_HCP7T/manifold_learning/Data_Interim/PNAS2015/{sbj}/LE```

import pandas as pd
import numpy as np
import os
import os.path as osp
import getpass
from datetime import datetime
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PRJ_DIR
from utils.basics import le_dist_metrics, le_knns, le_ms

# + [markdown] tags=[]
# ***
#
# The next cell select the Window Length ```wls``` and Window Step ```wss``` used to generate the matrices
# -

wls = 45
wss = 1.5

# ***
# Those are the scenarios we will be running

print('++ INFO: Distance Metrics: %s' % str(le_dist_metrics))
print('++ INFO: Knns:             %s' % str(le_knns))
print('++ INFO: Ms:               %s' % str(le_ms))

# The next cell will create the output folders if they do not exist already

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'LE')
    if not osp.exists(path):
        os.makedirs(path)

# The next cell will create folders for the swarm log files and for the actual swarm script. Those folders are created using the username as part of their name. That way it is easier for different users to work together on the project.

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N07_LE_Multitask.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N07_LE_Multitask.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
# -

# Create swarm script. This script will have one line per matrix to be generated.

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -b 20 -g 4 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    for dist in le_dist_metrics:
        for knn in le_knns:
            for m in le_ms:
                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'LE','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.pkl'.format(subject=subject,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                swarm_file.write('export path_tvfc={path_tvfc} dist={dist} knn={knn} m={m} path_out={path_out}; sh {scripts_dir}/N07_LE.sh'.format(path_tvfc=path_tvfc, 
                                                                                                                                    path_out=path_out,
                                                                                                                                    dist=dist,
                                                                                                                                    knn=knn,
                                                                                                                                    m=m, 
                                                                                                                                    scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()

# + [markdown] tags=[]
# ***
# # Run on Connection Randomization Model
# -

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'LE','Null_ConnRand')
    if not osp.exists(path):
        os.makedirs(path)

# The next cell will create folders for the swarm log files and for the actual swarm script. Those folders are created using the username as part of their name. That way it is easier for different users to work together on the project.

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N07_LE_Multitask_ConnRand.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N07_LE_Multitask_ConnRand.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
# -

# Create swarm script. This script will have one line per matrix to be generated.

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -b 20 -g 4 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    for dist in le_dist_metrics:
        for knn in le_knns:
            for m in le_ms:
                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'LE','Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.pkl'.format(subject=subject,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                swarm_file.write('export path_tvfc={path_tvfc} dist={dist} knn={knn} m={m} path_out={path_out}; sh {scripts_dir}/N07_LE.sh'.format(path_tvfc=path_tvfc, 
                                                                                                                                    path_out=path_out,
                                                                                                                                    dist=dist,
                                                                                                                                    knn=knn,
                                                                                                                                    m=m, 
                                                                                                                                    scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()

# + [markdown] tags=[]
# ***
# # Run on Phase Randomization Model
# -

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'LE','Null_PhaseRand')
    if not osp.exists(path):
        os.makedirs(path)

# The next cell will create folders for the swarm log files and for the actual swarm script. Those folders are created using the username as part of their name. That way it is easier for different users to work together on the project.

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N07_LE_Multitask_PhaseRand.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N07_LE_Multitask_PhaseRand.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
# -

# Create swarm script. This script will have one line per matrix to be generated.

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -b 20 -g 4 -t 4 --time 00:05:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    for dist in le_dist_metrics:
        for knn in le_knns:
            for m in le_ms:
                path_tvfc = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
                path_out  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'LE','Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.pkl'.format(subject=subject,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4)))
                swarm_file.write('export path_tvfc={path_tvfc} dist={dist} knn={knn} m={m} path_out={path_out}; sh {scripts_dir}/N07_LE.sh'.format(path_tvfc=path_tvfc, 
                                                                                                                                    path_out=path_out,
                                                                                                                                    dist=dist,
                                                                                                                                    knn=knn,
                                                                                                                                    m=m, 
                                                                                                                                    scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()
# -

