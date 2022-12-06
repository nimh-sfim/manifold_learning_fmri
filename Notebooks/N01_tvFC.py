import argparse
import numpy as np
import pandas as pd
from scipy.stats import zscore
from utils.data_functions import compute_SWC
from utils.null_models import randomize_conn, phase_randomize

def run(args):
    path_ints         = args.path_ints
    path_out_Z        = args.path_out_Z
    path_out_R        = args.path_out_R
    path_out_Z_normed = args.path_out_Z_normed
    path_out_R_normed = args.path_out_R_normed
    path_rois         = args.path_roi_names
    path_wins         = args.path_win_names
    wl_sec            = args.wls
    ws_sec            = args.wss
    tr_sec            = args.tr
    if args.null_model == 'none':
     null_model = None
    else:
     null_model = args.null_model
    print(' ')
    print('++ INFO: Run information')
    print(' +       Input Time series path     :', path_ints)
    print(' +       Ouput tvFC path (R)        :', path_out_R)
    print(' +       Ouput tvFC path (Z)        :', path_out_Z)
    print(' +       Ouput tvFC path (R) normed :', path_out_R_normed)
    print(' +       Ouput tvFC path (Z) normed :', path_out_Z_normed)
    print(' +       ROI Names file path        :', path_rois)
    print(' +       Window Names file path     :', path_wins)
    print(' +       TR [sec]                   :', tr_sec)
    print(' +       Window Length [sec]        :', wl_sec)
    print(' +       Window Step [sec]          :', ws_sec)
    print(' +       Null Model                 :', str(null_model))
    print(' ')
    
    # Read Window Names into memory
    win_names = np.loadtxt(path_wins, dtype='str')
    print(' + Number of window names: %d' % len(win_names))
    # Read ROI Names into memory
    roi_names = np.loadtxt(path_rois, dtype='str')
    print(' + Number of window names: %d' % len(roi_names))
    # Read ROI Timeseries into memory
    roi_ts = pd.read_csv(path_ints, sep='\t', header=None).T
    roi_ts.columns.name = 'ROI_Name'
    roi_ts.columns = roi_names
    roi_ts.index   = pd.timedelta_range(start='0',periods=roi_ts.shape[0],freq='{tr}L'.format(tr=tr_sec*1000))    
    # Apply Phase Randomization if requested
    if null_model == "phase_rand":
       print(" + WARNING: Phase Randomization requested")
       roi_ts = phase_randomize(roi_ts)
       
    # Compute Sliding Window Correlation
    wl_trs  = int(wl_sec/tr_sec)
    ws_trs  = int(ws_sec/tr_sec)
    print(' + Windo Information: WL = %d samples | WS = %d samples' % (wl_trs, ws_trs))
    swc_r,swc_Z, winInfo = compute_SWC(roi_ts,wl_trs,ws_trs,win_names=win_names,window=None)
    swc_r.index.name = 'Connections'
    swc_Z.index.name = 'Connections'
    print(" + Size of sliding window correlation: %s" % str(swc_r.shape))
    
    # Apply Connection Randomization if requested
    if null_model == "conn_rand":
       print(" + WARNING: Connection Randomization requested")
       print(" +          Randomizing swc_r...")
       swc_r = randomize_conn(swc_r.T).T
       print(" +          Randomizing swc_Z...")
       swc_Z = randomize_conn(swc_Z.T).T
    # Save to disk
    if path_out_R is not None:
       swc_r.to_pickle(path_out_R)
       print(' + tvFC (R) saved to disk: %s' % path_out_R)
    if path_out_Z is not None:
       swc_Z.to_pickle(path_out_Z)
       print(' + tvFC (Z) saved to disk: %s' % path_out_Z)
    # Zscoring
    if path_out_R_normed is not None:
       print('++ INFO: Z-scoring features in sw_r')
       swc_r = swc_r.apply(zscore,axis=1)
       swc_r.to_pickle(path_out_R_normed)
       print(' + tvFC (R) normed saved to disk: %s' % path_out_R_normed)
    if path_out_Z_normed is not None:
       print('++ INFO: Z-scoring features in sw_Z')
       swc_Z = swc_Z.apply(zscore,axis=1)
       swc_Z.to_pickle(path_out_Z_normed)
       print(' + tvFC (R) normed saved to disk: %s' % path_out_Z_normed)

def main():
    parser=argparse.ArgumentParser(description="Create tvFC matrix from a set of ROI timeseries")
    parser.add_argument("-ints",        help="Path to ROI timseries",           dest="path_ints",         type=str,   required=True)
    parser.add_argument("-outZ",        help="Path to output file (Z)",         dest="path_out_Z",        type=str,   required=False, default=None)
    parser.add_argument("-outR",        help="Path to output file (R)",         dest="path_out_R",        type=str,   required=False, default=None)
    parser.add_argument("-outZ_normed", help="Path to output file (Z) zscored", dest="path_out_Z_normed", type=str,   required=False, default=None)
    parser.add_argument("-outR_normed", help="Path to output file (R) zscored", dest="path_out_R_normed", type=str,   required=False, default=None)
    parser.add_argument("-roi_names",   help="Path to file with ROI names",     dest="path_roi_names",    type=str,   required=True)
    parser.add_argument("-win_names",   help="Path to file with Window names",  dest="path_win_names",    type=str,   required=True)
    parser.add_argument("-wls",         help="Window Length in seconds",        dest="wls",               type=float, required=True)
    parser.add_argument("-wss",         help="Window Step in seconds",          dest="wss",               type=float, required=True)
    parser.add_argument("-tr",          help="Repetition Time in seconds",      dest="tr",                type=float, required=True)
    parser.add_argument("-null_model",  help="Null Model to apply",             dest="null_model",        type=str,   required=True, choices=['none','conn_rand','phase_rand'])
    
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()