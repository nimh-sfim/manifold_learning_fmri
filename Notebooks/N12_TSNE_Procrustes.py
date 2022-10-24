import argparse
import numpy as np
import pandas as pd
from utils.random         import seed_value
from utils.basics         import PNAS2015_subject_list, PRJ_DIR
from utils.procrustes     import procrustes_scan_tsne_embs
from utils.io             import load_TSNE_SI
import os.path as osp
from sklearn.metrics import silhouette_score

def run(args):
    wls         = 45
    wss         = 1.5
    np.random.seed(seed_value)

    input_data  = args.input_data
    norm_method = args.norm_method
    dist        = args.dist
    pp          = args.pp
    m           = args.m
    alpha       = args.alpha
    init_method = args.init_method
    drop_xxxx   = bool(args.drop_xxxx)
    
    emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','TSNE',input_data,
                                'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3),alpha=str(alpha),
                                                                                                                      wss=str(wss),init=init_method,
                                                                                                                      dist=dist,
                                                                                                                      pp=str(pp).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
    si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','TSNE',input_data,
                               'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                      alpha=str(alpha),init=init_method,
                                                                                                                      wss=str(wss),
                                                                                                                      dist=dist,
                                                                                                                      pp=str(pp).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
    print(' ')
    print('++ INFO: Run information')
    print(' +       Seed Value              :', seed_value)
    print(' +       Input Data              :', input_data)
    print(' +       Normalization Method    :', norm_method)
    print(' +       Distance Function       :', dist)
    print(' +       Perplexity              :', pp)
    print(' +       Final Dimensionality    :', m)
    print(' +       Learning Rate           :', alpha)
    print(' +       Initialization Method   :', init_method)
    print(' +       Drop XXXX               :', drop_xxxx)
    print(' +       Procrusted Emb Out Path :', emb_path)
    print(' +       SI Out Path             :', si_path)
    print(' ')
    
    # Load SI for the embeddings to be aligned
    print('++ INFO: Loading Scan-level SI for all participanting scans...')
    si_TSNE = load_TSNE_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=False, wls=wls, wss=wss, 
                           input_datas=[input_data], norm_methods=[norm_method], dist_metrics=[dist],
                           init_methods=[init_method], pps=[pp], alphas=[alpha], ms=[m], no_tqdm=True)
    print(' + si_TSNE.shape = %s' % str(si_TSNE.shape))
    si_TSNE = si_TSNE.set_index(['Subject','Input Data','Norm','Metric','PP','m','Alpha','Init','Target']).sort_index()
    print(si_TSNE)
    
    # Compute Procrustes Transformation
    print('++ INFO: Computing Procrustes Transformation....')
    aux = procrustes_scan_tsne_embs(PNAS2015_subject_list,si_TSNE,input_data,norm_method,dist,pp,m,alpha, init_method,drop_xxxx=True)
    aux.index.name = 'WinID'
    aux.columns.name = ''
    print('++ INFO: Saving Group Level Embedding to disk...[%s]' % emb_path)
    aux.to_pickle(emb_path)
    
    # Evaluate the Quality of the Embedding with the SI index
    tsne_dims = [c for c in aux.columns if 'TSNE0' in c]
    print('++ INFO: Computing SIsbj...')
    si_sbj  = silhouette_score(aux[tsne_dims], aux.reset_index()['Subject'], n_jobs=-1)
    print(' +       SIsbj = %.2f' % si_sbj)
    print('++ INFO: Computing SItask...')
    si_task = silhouette_score(aux[tsne_dims], aux.reset_index()['Window Name'], n_jobs=-1)
    print(' +       SItask = %.2f' % si_task)
    
    # Writing SI results to disk
    df = pd.Series(index=['SI_Subject','SI_Window Name'], dtype=float)
    df['SI_Subject'] = si_sbj
    df['SI_Window Name'] = si_task
    print('++ INFO: Saving SI values to disk... [%s]' % si_path)
    df.to_pickle(si_path)
    
def main():
    parser=argparse.ArgumentParser(description="Create a Laplacian Eigenmap embedding given a tvFC matrix")
    parser.add_argument("-input_data",  help="Input data",                      dest="input_data",  type=str,  required=True)
    parser.add_argument("-norm_method", help="FC matrix normalization method",  dest="norm_method", type=str,  required=True)
    parser.add_argument("-dist",        help="Distance Metric",                 dest="dist",        type=str,  required=True)
    parser.add_argument("-pp",          help="Perplexity",                      dest="pp",          type=int,  required=True)
    parser.add_argument("-m",           help="Final Dimensionality",            dest="m",           type=int,  required=True)
    parser.add_argument("-alpha",       help="Learning Rate",                   dest="alpha",       type=int,  required=True)
    parser.add_argument("-init_method", help="Initialization Method",           dest="init_method", type=str,  required=True)
    parser.add_argument("-drop_xxxx",   help="Drop Task Inhomogenous Windows?", dest="drop_xxxx",    type=str,  required=True)

    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
