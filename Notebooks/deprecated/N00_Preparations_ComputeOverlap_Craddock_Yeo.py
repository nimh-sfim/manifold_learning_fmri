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

# ***
# ### Sort ROIs by Network

# 1. Download the Yeo Atlas from: ftp://surfer.nmr.mgh.harvard.edu/pub/data/Yeo_JNeurophysiol11_MNI152.zip
#
# 2. Separate into the different networks
#
# ```bash
# # cd /data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/Yeo_JNeurophysiol11_MNI152
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,1)' -prefix Yeo2011_7Networks_01_Visual.nii
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,2)' -prefix Yeo2011_7Networks_02_SomatoMotor.nii
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,3)' -prefix Yeo2011_7Networks_03_DorsalAttention.nii
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,4)' -prefix Yeo2011_7Networks_04_VentralAttention.nii
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,5)' -prefix Yeo2011_7Networks_05_Limbic.nii
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,6)' -prefix Yeo2011_7Networks_06_Control.nii
# 3dcalc -a Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -expr 'equals(a,7)' -prefix Yeo2011_7Networks_07_DMN.nii
# ```
#
# Now for each subject
#
#
# 3. Bring Networks into the same grid as the subject data
# ```bash
# # cd /data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/SBJ06
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_01_Visual.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_01_Visual
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_02_SomatoMotor.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_02_SomatoMotor
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_03_DorsalAttention.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_03_DorsalAttention
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_04_VentralAttention.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_04_VentralAttention
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_05_Limbic.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_05_Limbic
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_06_Control.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_06_Control
# 3dAllineate -1Dmatrix_apply /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SBJ06/D02_CTask001/SBJ06_CTask001.MNI2REF.Xaff12.1D -input ../../Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_07_DMN.nii -master SBJ06_Craddock_0200+orig. -final NN -prefix SBJ06.Yeo2011_7Networks_07_DMN
# ```
#
# 3. Run the script that computes the overlap between each Craddock ROI and each Yeo Network
#
# ```bash
# sh ./N00_Preparations_ComputeOveral_Craddock_Yeo.sh
# ```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.utils import Bunch
from nilearn.connectome import vec_to_sym_matrix
from utils.fc import load_netcc

overlap            = pd.read_csv('/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/SBJ06/SBJ06.Craddock2Yeo_Overlap.csv', index_col=[0])
overlap.head(5)

overlap['Network'] = [overlap.drop('Num_Voxels',axis=1).columns[overlap.drop('Num_Voxels',axis=1).loc[i].argmax()] for i in overlap.index]
overlap.sort_values(by='Network')

sorted_idxs = list(np.array(overlap.sort_values(by='Network').index) - 1)
sorted_rois = [ 'ROI{r}'.format(r=str(i).zfill(3)) for i in sorted_idxs]
sorted_net  = [overlap.loc[i+1]['Network'].split('_')[1] for i in sorted_idxs]

ROI_Sorting_Path = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/Other/Craddock_0200_SortByYeo7Networks.txt'

ROI_Sorting = pd.DataFrame(index=range(157),columns=['ROI_ID','Network_ID'])
ROI_Sorting['ROI_ID'] = sorted_rois
ROI_Sorting['Network_ID'] = sorted_net
ROI_Sorting.index.name='ROI Number'
ROI_Sorting.to_csv(ROI_Sorting_Path)

ROI_Sorting = pd.Series(sorted_rois)
ROI_Sorting.index.name = 'ROI_ID'
ROI_Sorting.name = 'ROI_Name'
ROI_Sorting.to_csv(ROI_Sorting_Path)

# ***
# ### Sanity Check #1 - Time series from MATLAB code (PNAS2015)

roi_ts = pd.read_csv('/data/SFIMJGC_HCP7T/manifold_learning_fmri/Resources/Figure01/sbj06_ctask001_nroi0200_wl030_ws001.csv', index_col=[0])

sns.heatmap(roi_ts[sorted_rois].corr(), cmap='RdBu_r', vmin=-.75, vmax=.75)

# ***
# ### Sanity Check #2 - tvFC matrix from MATLAB Code (PNAS2015)

print('++ INFO: Loading the tvFC dataset.....')
X_df = pd.read_csv('../Resources/Figure03/swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv.gz', index_col=[0,1])
# Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
X_df.columns = X_df.columns.str.split('.').str[0]

# Extract Task Lbaels (for coloring purposes)
labels             = pd.Series(X_df.columns)
X                  = X_df.values.T
X_orig             = X.copy()
(n_wins, n_conns)  = X.shape         # n = number of samples | d = original number of dimensions
print(' +       Input data shape = [%d Windows X %d Connections]' % (n_wins, n_conns))

# Replace zeros by a very small number to avoid division by zero errors
X[X==0] = 1e-12

# Convert to a sklearn.Bunch object
tvFC        = Bunch()
tvFC.data   = X
tvFC.labels = labels

# Plot the FC for one window as a sanity check
avg_matrix = vec_to_sym_matrix(X.mean(axis=0), diagonal=np.ones(157)/np.sqrt(2))

avg_matrix = pd.DataFrame(avg_matrix, 
                          index=['ROI{r}'.format(r=str(i).zfill(3)) for i in np.arange(157)],
                          columns = ['ROI{r}'.format(r=str(i).zfill(3)) for i in np.arange(157)])

# + tags=[]
sns.heatmap(avg_matrix.loc[sorted_rois,sorted_rois], cmap='RdBu_r',vmin=-1, vmax=1)
# -

# ***
# ### Sanity Check #3 - 3dNetCorr

netcc_path = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/SBJ06/SBJ06_Craddock_0200.WL180s_000.netcc'
sfc = load_netcc(netcc_path, roi_names=['ROI{r}'.format(r=str(i).zfill(3)) for i in np.arange(157)])

sns.heatmap(sfc.loc[sorted_rois,sorted_rois], cmap='RdBu_r',vmin=-1, vmax=1)

# # Regarding going back and forth between matrix and vector views

a=pd.DataFrame(np.array([[0,2,3,4,5],[2,0,6,7,8],[3,6,0,9,10],[4,7,9,0,11],[5,8,10,11,0]]))
a

a_nl_vec = sym_matrix_to_vec(a.values, discard_diagonal=True)
a_nl_vec

sel      = np.triu(np.ones(a.shape),1).astype(bool)
a_jv_vec = a.where(sel).T.stack().values
a_jv_vec

squareform(a.values)

vec_to_sym_matrix(a_nl_vec,diagonal=np.ones(5)/np.sqrt(2))

# + tags=[]
vec_to_sym_matrix(a_jv_vec,diagonal=np.ones(5)/np.sqrt(2))
