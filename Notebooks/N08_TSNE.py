import argparse
import numpy              as np
import pandas             as pd
from utils.data_functions import compute_SWC
from utils.random         import seed_value
from sklearn.utils        import Bunch, check_random_state
from sklearn.manifold     import TSNE

def run(args):
    path_tvfc   = args.path_tvfc
    path_out    = args.path_out
    dist        = args.dist
    pp          = args.pp
    m           = args.m
    lr          = args.lr
    n_iter      = args.n_iter
    init_method = args.init_method
    n_jobs      = args.n_jobs
    norm        = args.norm
    grad_method = args.grad_method
    bh_angle    = args.bh_angle
    print(' ')
    print('++ INFO: Run information')
    print(' +       Input path            :', path_tvfc)
    print(' +       Ouput path            :', path_out)
    print(' +       Distance Function     :', dist)
    print(' +       Perplexity            :', pp)
    print(' +       m                     :', m)
    print(' +       Learning Rate         :', lr)
    print(' +       Num Jobs              :', n_jobs)
    print(' +       Random Seed           :', seed_value)
    print(' +       Max. Num Iters        :', n_iter)
    print(' +       Init Method           :', init_method)
    print(' +       Feature Normalization :', norm)
    print(' +       Gradient Method       :', grad_method)
    if grad_method == 'barnes_hut':
         print(' +       Barnes-Hut Angle      :', bh_angle)
    print(' ')
    
    # Read tvFC matrix
    # ================
    tvFC = pd.read_pickle(path_tvfc)
    tvFC[tvFC==0]            = 1e-12
    print(" + tvFC shape: %s" % str(tvFC.shape))
    # Extract labels
    labels = tvFC.columns
    print(' + Labels    : %s' % str(labels[0:5]))
   
    # Normalize Features
    # ==================
    if norm == 'zscore':
       print(" + FEATURE NORMALIZATION IS ON ==> Z-score each feature separately")
       tvFC = tvFC.apply(zscore,axis=1)
 
    # Call the TSNE Function
    # ======================
    random_state = check_random_state(seed_value)
    TSNE_obj     = TSNE(n_components=m, 
                        perplexity=pp, 
                        metric=dist,
                        n_jobs=n_jobs,
                        init=init_method, 
                        random_state=random_state, 
                        method=grad_method,angle=bh_angle, 
                        learning_rate=lr, verbose=2, 
                        n_iter=n_iter)
    TSNE_emb = TSNE_obj.fit_transform(tvFC.T)
    
    # Save Embedding into DataFrame Object
    # ====================================
    dim_labels      = ['TSNE'+str(i+1).zfill(3) for i in range(m)]
    print(' + Dataframe Columns: %s' % str(dim_labels))
    df              = pd.DataFrame(TSNE_emb, columns=dim_labels)
    df.columns.name = 'TSNE dimensions'
    df.index        = labels
    df.index.name   = 'Window Name'
    print(' + About to save results to disk.....') 
    # Save output to disk
    # ===================
    df.to_pickle(path_out)
    print(' + TSNE result saved to disk: %s' % path_out)
    print('++ PROGRAM ENDED SUCCESSFULLY.')
def main():
    parser=argparse.ArgumentParser(description="Create a T-SNE embedding given a tvFC matrix")
    parser.add_argument("-tvfc",        help="Path to tvFC matrix",     dest="path_tvfc",   type=str,  required=True)
    parser.add_argument("-out",         help="Path to output file",     dest="path_out",    type=str,  required=True)
    parser.add_argument("-dist",        help="Distance function",       dest="dist",        type=str,  required=True)
    parser.add_argument("-pp",          help="Perplexity",              dest="pp",          type=int,  required=True)
    parser.add_argument("-m",           help="Number of dimensions",    dest="m",           type=int,  required=True)
    parser.add_argument("-lr",          help="Learning Rate",           dest="lr",          type=float,required=True)
    parser.add_argument("-n_iter",      help="Max. Number Iterations",  dest="n_iter",      type=int,  required=True)
    parser.add_argument("-n_jobs",      help="Number of jobs",          dest="n_jobs",      type=int,  required=True)
    parser.add_argument("-init",        help="Init Method",             dest="init_method", type=str,  required=True)
    parser.add_argument("-norm",        help="Normalize Features",      dest="norm",        type=str,  required=True)
    parser.add_argument("-grad_method", help="Gradient Descent Method", dest="grad_method", type=str,  required=False, default='exact', choices=['exact','barnes_hut']) 
    parser.add_argument("-bh_angle",    help="Barnes-Hut Angle",        dest="bh_angle",    type=float,required=False, default=0.5)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
