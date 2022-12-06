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

# # DESCRIPTION: Create Connectivity Matrices with netcc for all subjects
#
# This data has been previously used in other projects, and therefore it already exists locally in pre-processed form. Here for the purpose of this work, we will re-compute representative time-series per ROI of the Craddock 200 ROI atlas using AFNI's command 3dNetCorr. The output of running this command will be saved in ```/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/```. Anybody downloading this repo should be able to start from the next notebook using the files available in there.

from utils.basics import PNAS2015_folder, PNAS2015_subject_list
import subprocess
import os.path as osp
import os

# Define output folder

target_folder = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/'

# Create one folder per subject

for sbj in PNAS2015_subject_list:
    target_folder_sbj = osp.join(target_folder,sbj)
    if not osp.exists(target_folder_sbj):
        os.makedirs(target_folder_sbj)

# For each scan, run 3dNetCorr and extract time series.

# + tags=[]
# %%time
for sbj in PNAS2015_subject_list:
    target_folder_sbj = osp.join(target_folder,sbj)
    command = """module load afni; \
                 cd {target_folder_sbj}; \
                 3dcalc -overwrite -datum short -a {PNAS2015_folder}/PrcsData/{sbj}/D02_CTask001/{sbj}_CTask001.Craddock_T2Level_0200.lowSigma+orig.HEAD -expr 'a*(l+1)' -prefix {sbj}_Craddock_0200; \
                 3dTstat -overwrite -max -prefix {sbj}_Craddock_0200 {sbj}_Craddock_0200+orig; \
                 echo "++ INFO: Creating TS and FC for tvFC with 45s windows [0.023 - 0.18 Hz]"; \
                 3dNetCorr -overwrite -ts_out -in_rois {sbj}_Craddock_0200+orig -mask  {PNAS2015_folder}/PrcsData/{sbj}/D02_CTask001/pb06.{sbj}_CTask001.bpf.WL045.mask.lowSigma+orig.HEAD -inset {PNAS2015_folder}/PrcsData/{sbj}/D02_CTask001/pb08.{sbj}_CTask001.blur.WL045+orig.HEAD -prefix {sbj}_Craddock_0200.WL045s; \
                 """.format(target_folder_sbj=target_folder_sbj,
                         PNAS2015_folder=PNAS2015_folder,
                         sbj=sbj)
    output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print(output.strip().decode())
    print('++ ========================================================================================= ++')
