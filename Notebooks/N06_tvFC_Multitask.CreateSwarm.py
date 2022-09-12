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
import os
import os.path as osp
import getpass
from datetime import datetime
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

# The next cell will create the output folders if they do not exist already

# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject)
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
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --time 00:30:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for subject in PNAS2015_subject_list:
    path_ints  = osp.join(PRJ_DIR,'Data','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s_000.netts'.format(subject=subject,wls=str(int(wls)).zfill(3)))
    path_out_R = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    swarm_file.write('export path_roits={path_rois} path_roinames={path_roinames}  path_winnames={path_winnames} out_Z={path_out_Z} out_R={path_out_R} wls={wls} wss={wss} tr={tr} null=none; sh {scripts_dir}/N06_tvFC.sh'.format(
                       path_rois=path_ints, path_roinames=PNAS2015_roi_names_path, path_winnames=win_names_path,
                       path_out_Z=path_out_Z, path_out_R=path_out_R, wls=str(wls), wss=str(wss), tr=str(tr), 
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
    swarm_file.write('\n')
swarm_file.close()
# -
# ***
#
# # Null Model 1 - Connection Randomization


# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand')
    if not osp.exists(path):
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
    path_out_R = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_ConnRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    swarm_file.write('export path_roits={path_rois} path_roinames={path_roinames}  path_winnames={path_winnames} out_Z={path_out_Z} out_R={path_out_R} wls={wls} wss={wss} tr={tr} null=conn_rand; sh {scripts_dir}/N06_tvFC.sh'.format(
                       path_rois=path_ints, path_roinames=PNAS2015_roi_names_path, path_winnames=win_names_path,
                       path_out_Z=path_out_Z, path_out_R=path_out_R, wls=str(wls), wss=str(wss), tr=str(tr), 
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
    swarm_file.write('\n')
swarm_file.close()
# -

# ***
#
# # Null Model 2 - Phase Randomization


# Create Output Folders if they do not exists
for subject in PNAS2015_subject_list:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand')
    if not osp.exists(path):
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
    path_out_R = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.R.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    path_out_Z = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',subject,'Null_PhaseRand','{subject}_Craddock_0200.WL{wls}s.WS{wss}s.tvFC.Z.pkl'.format(subject=subject,wls=str(int(wls)).zfill(3), wss=str(wss)))
    swarm_file.write('export path_roits={path_rois} path_roinames={path_roinames}  path_winnames={path_winnames} out_Z={path_out_Z} out_R={path_out_R} wls={wls} wss={wss} tr={tr} null=phase_rand; sh {scripts_dir}/N06_tvFC.sh'.format(
                       path_rois=path_ints, path_roinames=PNAS2015_roi_names_path, path_winnames=win_names_path,
                       path_out_Z=path_out_Z, path_out_R=path_out_R, wls=str(wls), wss=str(wss), tr=str(tr), 
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
    swarm_file.write('\n')
swarm_file.close()
# -




