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
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PNAS2015_roi_names_path, PNAS2015_win_names_paths, PRJ_DIR

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

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original')
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

swarm_path     = osp.join(swarm_folder,'N06_tvFC_Multitask.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N06_tvFC_Multitask.logs')

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
swarm_file.write('#swarm -f {swarm_path} -b 2 -g 32 -t 32 --time 00:30:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    path_ints         = osp.join(PRJ_DIR,'Data','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s_000.netts'.format(subject=subject,wls=str(int(wls)).zfill(3)))
    path_out_R        = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.asis.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z        = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.asis.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_R_normed = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.zscored.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z_normed = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Original','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.zscored.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    swarm_file.write('export path_roits={path_rois} path_roinames={path_roinames}  path_winnames={path_winnames} out_Z={path_out_Z} out_R={path_out_R} out_Z_normed={path_out_Z_normed} out_R_normed={path_out_R_normed} wls={wls} wss={wss} tr={tr} null=none; sh {scripts_dir}/N06_tvFC.sh'.format(
                       path_rois=path_ints, path_roinames=PNAS2015_roi_names_path, path_winnames=win_names_path,
                       path_out_Z=path_out_Z, path_out_R=path_out_R, 
                       path_out_Z_normed=path_out_Z_normed, path_out_R_normed=path_out_R_normed,
                       wls=str(wls), wss=str(wss), tr=str(tr), 
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
    swarm_file.write('\n')
swarm_file.close()
# -
# ## 1.2 Null Model - Connection Randomization


# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand')
    if not osp.exists(path):
        print('++ INFO: Creating folder %s' % path)
        os.makedirs(path)

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N06_tvFC_Multitask_Null_ConnRand.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N06_tvFC_Multitask_Null_ConnRand.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --time 00:30:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    path_ints  = osp.join(PRJ_DIR,'Data','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s_000.netts'.format(subject=subject,wls=str(int(wls)).zfill(3)))
    path_out_R = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.asis.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.asis.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_R_normed = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.zscored.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z_normed = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.zscored.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    
    swarm_file.write('export path_roits={path_rois} path_roinames={path_roinames}  path_winnames={path_winnames} out_Z={path_out_Z} out_R={path_out_R} out_Z_normed={path_out_Z_normed} out_R_normed={path_out_R_normed} wls={wls} wss={wss} tr={tr} null=conn_rand; sh {scripts_dir}/N06_tvFC.sh'.format(
                       path_rois=path_ints, path_roinames=PNAS2015_roi_names_path, path_winnames=win_names_path, 
                       path_out_Z=path_out_Z, path_out_R=path_out_R, wls=str(wls), wss=str(wss), tr=str(tr), path_out_Z_normed=path_out_Z_normed, path_out_R_normed=path_out_R_normed,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
    swarm_file.write('\n')
swarm_file.close()
# -

# ## 1.3 Null Model - Phase Randomization


# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand')
    if not osp.exists(path):
        print('++ INFO: Creating folder %s' % path)
        os.makedirs(path)

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N06_tvFC_Multitask_Null_PhaseRand.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N06_tvFC_Multitask_Null_PhaseRand.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --time 00:30:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    path_ints  = osp.join(PRJ_DIR,'Data','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s_000.netts'.format(subject=subject,wls=str(int(wls)).zfill(3)))
    path_out_R = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.asis.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.asis.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_R_normed = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.zscored.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z_normed = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.zscored.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    
    swarm_file.write('export path_roits={path_rois} path_roinames={path_roinames}  path_winnames={path_winnames} out_Z={path_out_Z} out_R={path_out_R} out_Z_normed={path_out_Z_normed} out_R_normed={path_out_R_normed} wls={wls} wss={wss} tr={tr} null=phase_rand; sh {scripts_dir}/N06_tvFC.sh'.format(
                       path_rois=path_ints, path_roinames=PNAS2015_roi_names_path, path_winnames=win_names_path,
                       path_out_Z=path_out_Z, path_out_R=path_out_R, wls=str(wls), wss=str(wss), tr=str(tr), path_out_Z_normed=path_out_Z_normed, path_out_R_normed=path_out_R_normed,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
    swarm_file.write('\n')
swarm_file.close()
# -
# ***
# # 2. Group-level Matrices


group_tvFCs = {('Original','asis'):pd.DataFrame(),   ('Null_ConnRand','asis'):pd.DataFrame(),   ('Null_PhaseRand','asis'):pd.DataFrame(),
               ('Original','zscored'):pd.DataFrame(),('Null_ConnRand','zscored'):pd.DataFrame(),('Null_PhaseRand','zscored'):pd.DataFrame(),}

# %%time
for scenario in tqdm(['asis','zscored'], desc='Scenario',leave=False):
    for sbj in tqdm(PNAS2015_subject_list,desc='Subjects'):
        for data_input in ['Original','Null_ConnRand','Null_PhaseRand']:
            tvFC_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,data_input,'{sbj}_Craddock_0200.WL045s.WS1.5s.tvFC.Z.{scenario}.pkl'.format(sbj=sbj, scenario=scenario))
            tvFC = pd.read_pickle(tvFC_path)
            group_tvFCs[data_input,scenario] = pd.concat([group_tvFCs[data_input,scenario],tvFC],axis=1)

for data_input in ['Original','Null_ConnRand','Null_PhaseRand']:
    out_dir = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL',data_input)
    if not osp.exists(out_dir):
        print("+ Create output folder: %s" % out_dir)
        os.makedirs(out_dir)

# Before saving to disk, we will add the subject and task info to the colum of the dataframe

[N_cons, N_wins]=tvFC.shape
sbj_labels      = []
for s in PNAS2015_subject_list:
    sbj_labels = sbj_labels + list(np.tile(s,N_wins))
win_labels = group_tvFCs['Original','asis'].columns
column_names = pd.MultiIndex.from_arrays([sbj_labels,win_labels],names=['Subject','Window Name'])

for data_input in ['Original','Null_ConnRand','Null_PhaseRand']:
    for scenario in ['asis','zscored']:
        group_tvFCs[data_input,scenario].columns    = column_names
        group_tvFCs[data_input,scenario].index.name = 'Connections'

# %%time
for data_input in ['Original','Null_ConnRand','Null_PhaseRand']:
    for scenario in ['asis','zscored']:
        out_dir = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','ALL',data_input)
        group_tvFC_path = osp.join(out_dir,'ALL_Craddock_0200.WL045s.WS1.5s.tvFC.Z.{scenario}.pkl'.format(scenario=scenario))
        group_tvFCs[data_input,scenario].to_pickle(group_tvFC_path)
        print('++ INFO: Size of [%s,%s] Group-level Matrix [%s] | Save to %s' % (data_input,scenario,str(group_tvFCs[data_input,scenario].shape),group_tvFC_path))


