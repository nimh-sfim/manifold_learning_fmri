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
# This notebook will compute sliding window correlation matrices for the 20 subjects in the multi-task dataset.
#
# Initially, we are only working with WL = 45s and WS = 1.5s.
#
# Matrices will be written as pandas pickle objects in ```/data/SFIMJGC_HCP7T/manifold_learning/Data_Interim/PNAS2015```

import pandas as pd
import numpy as np
import os
import os.path as osp
import getpass
from datetime import datetime
from tqdm.notebook import tqdm
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PNAS2015_roi_names_path, PNAS2015_win_names_paths, PRJ_DIR, input_datas, norm_methods

# + [markdown] tags=[]
# ***
#
# The next cell select the Window Length ```wls``` and Window Step ```wss``` used to generate the matrices
# -

wls = 45
wss = 1.5
tr  = 1.5
win_names_path = PNAS2015_win_names_paths[(wls,wss)]

# ***
#
# # 1. Scan-Level Matrices
#
# ## 1.1. Original Data
#
# The next cell will create the output folders if they do not exist already

# + tags=[]
# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'ID_estimates',input_data)
        if not osp.exists(path):
            print('++ INFO: Created folder %s' % path)
            os.makedirs(path)
# -

# The next cell will create folders for the swarm log files and for the actual swarm script. Those folders are created using the username as part of their name. That way it is easier for different users to work together on the project.

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N14_ID_estimates.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N14_ID_estimates.logs')

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
swarm_file.write('#swarm -J ID_estimates -f {swarm_path} -g 32 -t 8 --time 03:00:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        for norm_method in norm_methods:
            tvFC_path        = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
            out_path_local   = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'ID_estimates',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.local_ID.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
            out_path_global  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'ID_estimates',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.global_ID.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))

            swarm_file.write('export tvfc_path={tvFC_path}  out_path_local={out_path_local} out_path_global={out_path_global} n_jobs=4; sh {scripts_dir}/N14_ID.sh'.format(
                       tvFC_path = tvFC_path, out_path_local=out_path_local, out_path_global=out_path_global,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
            swarm_file.write('\n')
swarm_file.close()
# -
# ***

needed_global = 0
avail_global = 0
for subject in PNAS2015_subject_list:
    for input_data in input_datas:
        for norm_method in norm_methods:
            needed_global += 1
            out_path_global  = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'ID_estimates',input_data,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.{nm}.global_ID.pkl'.format(subject=subject,nm=norm_method,wls=str(int(wls)).zfill(3), wss=str(wss)))
            if osp.exists(out_path_global):
                avail_global += 1
            else:
                print(out_path_global)
print('++ INFO: Files avail/needed [%d/%d]' %(avail_global,needed_global))


