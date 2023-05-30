import argparse
import numpy as np
import pandas as pd
from utils.data_functions import compute_SWC
from utils.random         import seed_value
from sklearn.utils        import Bunch, check_random_state
import umap

def run(args):
    path_tvfc = args.path_tvfc
    path_out  = args.path_out
    dist      = args.dist
    knn       = args.knn
    m         = args.m
    alpha     = args.alpha 
    min_dist  = args.min_dist
    init      = args.init
    
    print(' ')
    print('++ INFO: Run information')
    print(' +       Input path        :', path_tvfc)
    print(' +       Ouput path        :', path_out)
    print(' +       Distance Function :', dist)
    print(' +       Knn               :', knn)
    print(' +       m                 :', m)
    print(' +       Learning Rate     :', alpha)
    print(' +       Random Seed       :', seed_value)
    print(' +       Minimum Distance  :', min_dist)
    print(' +       Init Method.      :', init)
    print(' ') 
   
    # Read tvFC matrix
    # ================
    tvFC = pd.read_pickle(path_tvfc)
    tvFC[tvFC==0]            = 1e-12
    print(" + tvFC shape: %s" % str(tvFC.shape))
    # Extract labels
    labels = tvFC.columns
    print(' + Labels    : %s' % str(labels[0:5]))
    
    # Call the UMAP Function
    # ======================
    if args.use_random_seed == True:
       print("++ WARNING: Using a random seed (Stability Run)")
       random_state = check_random_state(None)
    else:
       random_state = check_random_state(seed_value)
    print(' + Random State created....')
   
    print(' + Creating UMAP object....')
    mapper = umap.UMAP(n_neighbors=knn, 
                       min_dist=min_dist, 
                       metric=dist, 
                       random_state=random_state, 
                       verbose=True, 
                       n_components=m,
                       init=init,
                       learning_rate=alpha, n_jobs=2,
                       n_epochs=1000)
    print(' + Running UMAP.fit function...')
    mapper.fit(tvFC.T.values)
    print(' + UMAP.fit finished.')
    
    # Save Embedding into DataFrame Object
    # ====================================
    dim_labels      = ['UMAP'+str(i+1).zfill(3) for i in range(m)]
    print(' + Dataframe Columns: %s' % str(dim_labels))
    df              = pd.DataFrame(mapper.embedding_, columns=dim_labels)
    df.columns.name = 'UMAP dimensions'
    df.index        = labels
    df.index.name   = 'Window Name'
    
    # Save output to disk
    # ===================
    df.to_pickle(path_out)
    print(' + UMAP result saved to disk: %s' % path_out)

def main():
    parser=argparse.ArgumentParser(description="Create a UMAP embedding given a tvFC matrix")
    parser.add_argument("-tvfc",     help="Path to tvFC matrix",   dest="path_tvfc", type=str,   required=True)
    parser.add_argument("-out",      help="Path to output file",   dest="path_out",  type=str,   required=True)
    parser.add_argument("-dist",     help="Distance function",     dest="dist",      type=str,   required=True)
    parser.add_argument("-knn",      help="Neighborhood size",     dest="knn",       type=int,   required=True)
    parser.add_argument("-m",        help="Number of dimensions",  dest="m",         type=int,   required=True)
    parser.add_argument("-alpha",    help="Initial Learning Rate", dest="alpha",     type=float, required=True)
    parser.add_argument("-min_dist", help="Minimum Distance",      dest="min_dist",  type=float, required=True)
    parser.add_argument("-init",     help="Init Method",           dest="init",      type=str,   required=True)
    parser.add_argument("-random_seed", help="Use a random seed",    dest="use_random_seed", required=False, action='store_true')
    parser.set_defaults(use_random_seed=False)
    
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
