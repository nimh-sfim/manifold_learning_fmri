import argparse
import numpy as np
import pandas as pd
from utils.data_functions import compute_SWC
from utils.random         import seed_value
from sklearn.utils        import Bunch, check_random_state
import skdim
from tqdm.auto import tqdm

def run(args):
    path_tvfc       = args.path_tvfc
    path_out_local  = args.path_out_local
    path_out_global = args.path_out_global
    n_jobs          = args.n_jobs
    knn_list        = [25,50,75,100,125,150,200]
    print(' ')
    print('++ INFO: Run information')
    print(' +       Input path              :', path_tvfc)
    print(' +       Ouput path  [Local ID]  :', path_out_local)
    print(' +       Ouput path  [Global ID] :', path_out_global)
    print(' +       # Jobs                  :', n_jobs)
    print(' +       Random Seed             :', seed_value)
    print(' +       Knns                    :', knn_list)
    print(' ')
    
    # Read tvFC matrix
    # ================
    tvFC = pd.read_pickle(path_tvfc)
    print(" + tvFC shape: %s" % str(tvFC.shape))
    # Extract labels
    labels = tvFC.columns
    print(' + Labels    : %s' % str(labels[0:5]))
    
    # Prepare output DataFrame
    # ========================
    lpca_cols             = ['lpca_'+str(knn) for knn in knn_list]
    twoNN_cols            = ['twoNN_'+str(knn) for knn in knn_list]
    fisherS_cols          = ['fisherS_'+str(knn) for knn in knn_list]
    out_df                = pd.DataFrame(columns=['Window Name']+lpca_cols+twoNN_cols+fisherS_cols)
    out_df['Window Name'] = labels
    
    # Compute local ID with Local PCA
    # ===============================
    print('++ INFO: Computing local ID with LPCA')
    for knn in tqdm(knn_list, desc='Local PCA:'):
        lpca_id_estimator = skdim.id.lPCA()
        lpca_ids = lpca_id_estimator.fit_transform_pw(tvFC.T,n_jobs=n_jobs, n_neighbors=knn)
        out_df['lpca_'+str(knn)] = lpca_ids
       
    # Compute local ID with twoNN
    # ===========================
    print('++ INFO: Computing local ID with TwoNN')
    for knn in tqdm(knn_list, desc='TwoNN:'):
        twoNN_id_estimator = skdim.id.TwoNN()
        twoNN_ids = twoNN_id_estimator.fit_transform_pw(tvFC.T,n_jobs=n_jobs, n_neighbors=knn)
        out_df['twoNN_'+str(knn)] = twoNN_ids
    
    
    # Compute local ID with FisherS
    # =============================
    print('++ INFO: Computing local ID with FisherS')
    for knn in tqdm(knn_list, desc='FisherS:'):
        fisherS_id_estimator = skdim.id.FisherS()
        fisherS_ids = fisherS_id_estimator.fit_transform_pw(tvFC.T,n_jobs=n_jobs, n_neighbors=knn)
        out_df['fisherS_'+str(knn)] = fisherS_ids
        
    print('++ INFO: Final Local ID DataFrame Size = %s' % str(out_df.shape))
    
    # Write Local ID to disk
    # ======================
    out_df.to_pickle(path_out_local)
    print('++ INFO: Local ID saved to disk [%s]' % path_out_local)
    
    # Compute GlobalID
    # ================
    print('++ INFO: Computing global ID with LPCA')
    lpca_id_estimator = skdim.id.lPCA()
    lpca_gID = lpca_id_estimator.fit_transform(tvFC.T)
    print('++ INFO: Computing global ID with TwoNN')
    twoNN_id_estimator = skdim.id.TwoNN()
    twoNN_gID = twoNN_id_estimator.fit_transform(tvFC.T)
    print('++ INFO: Computing global ID with FisherS')
    fisherS_id_estimator = skdim.id.FisherS()
    fisherS_gID = fisherS_id_estimator.fit_transform(tvFC.T)
    
    # Write Global ID results to disk
    # ===============================
    out_df2 = pd.Series([lpca_gID,twoNN_gID,fisherS_gID],index=['lpca_global','twoNN_global','fisherS_global'], name='globalID')
    out_df2.to_pickle(path_out_global)
    print('++ INFO: Global ID saved to disk [%s]' % path_out_global)


def main():
    parser=argparse.ArgumentParser(description="Create a Laplacian Eigenmap embedding given a tvFC matrix")
    parser.add_argument("-tvfc",   help="Path to tvFC matrix",  dest="path_tvfc", type=str,  required=True)
    parser.add_argument("-out_local",    help="Path to output file with local IDs",  dest="path_out_local",  type=str,  required=True)
    parser.add_argument("-out_global",    help="Path to output file with global IDs",  dest="path_out_global",  type=str,  required=True)
    parser.add_argument("-n_jobs", help="Number of Jobs",       dest="n_jobs",    type=int,  required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
