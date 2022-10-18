import argparse
import numpy as np
import pandas as pd
from utils.data_functions import compute_SWC
from utils.random         import seed_value
from sklearn.utils        import Bunch, check_random_state
from sklearn.manifold     import SpectralEmbedding
from sklearn.neighbors    import kneighbors_graph

def run(args):
    path_tvfc = args.path_tvfc
    path_out  = args.path_out
    dist      = args.dist
    knn       = args.knn
    m         = args.m
    print(' ')
    print('++ INFO: Run information')
    print(' +       Input path        :', path_tvfc)
    print(' +       Ouput path        :', path_out)
    print(' +       Distance Function :', dist)
    print(' +       Knn               :', knn)
    print(' +       m                 :', m)
    print(' +       Random Seed       :', seed_value)
    print(' ')
    
    # Read tvFC matrix
    # ================
    tvFC = pd.read_pickle(path_tvfc)
    print(" + tvFC shape: %s" % str(tvFC.shape))
    # Extract labels
    labels = tvFC.columns
    print(' + Labels    : %s' % str(labels[0:5]))
    
    # Compute Affinity matrix
    # =======================
    # NOTE: inclue_self is True for connectivity by default in sklearn. I think for us it makes no sense
    # to have 1.0 in the diagonal here... but should check.
    Xaff_non_symmetric = pd.DataFrame(kneighbors_graph(tvFC.T, 
                                               n_neighbors  = knn, 
                                               include_self = False, 
                                               n_jobs       = -1, 
                                               metric       = dist, 
                                               mode='connectivity').toarray())
    
    # Symmetrization is done as in sklearn (for compatibility) 
    Xaff = 0.5 * (Xaff_non_symmetric + Xaff_non_symmetric.T)
    # If we wanted to implement the original description by Belkin et al. we would use the following:
    # Belkin et al.: Xaff = ((0.5 * (Xaff_non_symmetric + Xaff_non_symmetric.T)) > 0).astype(int)
    print(" + Affinity matrix shape: %s" % str(Xaff.shape))
    
    # Create Spectral Embedding Object
    # ================================
    print(' + Create Spectral Embedding Object....')
    random_state = check_random_state(seed_value)
    LE_obj       = SpectralEmbedding(n_components=m, affinity='precomputed', n_jobs=-1, random_state=random_state)
    
    # Compute Embedding
    # =================
    print(' + Compute the Spectral Embeddding (call to fit_transform)...')
    le           = LE_obj.fit_transform(Xaff)
    
    # Compose Output DataFrame Object
    # ===============================
    print(' + Making final DataFrame with embedding results...')
    dim_labels      = ['LE'+str(i+1).zfill(3) for i in range(m)]
    df              = pd.DataFrame(le, columns=dim_labels)
    df.columns.name = 'LE dimensions'
    df.index        = labels
    df.index.name   = 'Window Name'

    # Save output to disk
    # ===================
    print(' + Saving embedding to disk...')
    df.to_pickle(path_out)
    print(' + LE saved to disk: %s' % path_out)
    
def main():
    parser=argparse.ArgumentParser(description="Create a Laplacian Eigenmap embedding given a tvFC matrix")
    parser.add_argument("-tvfc",   help="Path to tvFC matrix",  dest="path_tvfc", type=str,  required=True)
    parser.add_argument("-out",    help="Path to output file",  dest="path_out",  type=str,  required=True)
    parser.add_argument("-dist",   help="Distance function",    dest="dist",      type=str,  required=True)
    parser.add_argument("-knn",    help="Neighborhood size",    dest="knn",       type=int,  required=True)
    parser.add_argument("-m",      help="Number of dimensions", dest="m",         type=int,  required=True)
    
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
