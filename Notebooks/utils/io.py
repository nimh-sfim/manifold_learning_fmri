import pandas as pd
import numpy as np

def read_afni_roidat(path, verb=True):
    roi_info = pd.read_csv(path, engine='python', skipinitialspace=True, sep=' ', skiprows=1, header=None, names=['Nv','Nnnv','Frac','X','ROI','ROI_Orig_ID']).drop('X',axis=1).set_index('ROI')
    roi_info['Hemisphere'] = [r.split('_')[1] for r in roi_info['ROI_Orig_ID'] ]
    roi_info['Network']    = [r.split('_')[2] for r in roi_info['ROI_Orig_ID'] ]
    roi_info['ROI']        = ['_'.join(r.split('_')[3:]) for r in roi_info['ROI_Orig_ID'] ]
    print("++ INFO [read_afni_roidat]: Number of ROIs = %d " % roi_info.shape[0])
    return roi_info