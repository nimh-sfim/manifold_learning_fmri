import argparse
import numpy as np
import pandas as pd
from utils.random         import seed_value
from utils.basics         import PNAS2015_subject_list, PRJ_DIR
from utils.procrustes     import procrustes_scan_umap_embs
from utils.io             import load_UMAP_SI
import os.path as osp
from sklearn.metrics import silhouette_score

def run(args):
    wls         = 45
    wss         = 1.5
    np.random.seed(seed_value)

    sbj_list    = args.sbj_list.split(',')
    input_data  = args.input_data
    norm_method = args.norm_method
    dist        = args.dist
    knn         = args.knn
    m           = args.m
    alpha       = args.alpha
    init_method = args.init_method
    mdist       = args.mdist
    if args.drop_xxxx == "True":
       drop_xxxx = True
    elif args.drop_xxxx == "False":
       drop_xxxx = False
    
    emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,
                                'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{mdist}_a{alpha}_{init}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3),alpha=str(alpha),
                                                                                                                      wss=str(wss),init=init_method,
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      nm=norm_method,
                                                                                                                      mdist=str(mdist),
                                                                                                                      m=str(m).zfill(4)))
    si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','UMAP',input_data,
                               'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{mdist}_a{alpha}_{init}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                      alpha=str(alpha),init=init_method,
                                                                                                                      wss=str(wss),
                                                                                                                      dist=dist,
                                                                                                                      knn=str(knn).zfill(4),
                                                                                                                      mdist=str(mdist),
                                                                                                                      nm=norm_method,
                                                                                                                      m=str(m).zfill(4)))
    print(' ')
    print('++ INFO: Run information')
    print(' +       Seed Value              :', seed_value)
    print(' +       Input Data              :', input_data)
    print(' +       Normalization Method    :', norm_method)
    print(' +       Distance Function       :', dist)
    print(' +       Neighborhood Size       :', knn)
    print(' +       Final Dimensionality    :', m)
    print(' +       Learning Rate           :', alpha)
    print(' +       Initialization Method   :', init_method)
    print(' +       Minimum Distance        :', mdist)
    print(' +       Drop XXXX               :', drop_xxxx)
    print(' +       Procrusted Emb Out Path :', emb_path)
    print(' +       SI Out Path             :', si_path)
    print(' ')
    
    # Load SI for the embeddings to be aligned
    print('++ INFO: Loading Scan-level SI for all participanting scans...')
    si_UMAP = load_UMAP_SI(sbj_list=PNAS2015_subject_list,check_availability=False, verbose=False, wls=wls, wss=wss, 
                           input_datas=[input_data], norm_methods=[norm_method], dist_metrics=[dist],
                           init_methods=[init_method], knns=[knn], alphas=[alpha], ms=[m], no_tqdm=True)
    print(' + si_UMAP.shape = %s' % str(si_UMAP.shape))
    print(' + si_UMAP.head(5)')
    print(si_UMAP.head(5))
    print(' + =========================================')
    si_UMAP = si_UMAP.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Alpha','Init','MinDist','Target']).sort_index()
    print(si_UMAP)
    
    # Compute Procrustes Transformation
    print('++ INFO: Computing Procrustes Transformation....')
    aux = procrustes_scan_umap_embs(PNAS2015_subject_list,si_UMAP,input_data,norm_method,dist,knn,alpha, init_method,mdist,m,drop_xxxx=drop_xxxx)
    if aux is None:
        print ('++ ERROR: Exiting program.')
        return None
    aux = aux.set_index(['Subject','Window Name'])
    aux.columns.name = 'UMAP Dimensions'
    #aux.index.name = 'WinID'
    #aux.columns.name = ''
    print('++ INFO: Saving Group Level Embedding to disk...[%s]' % emb_path)
    aux.to_pickle(emb_path)
    
    # Evaluate the Quality of the Embedding with the SI index
    # Evaluation is always only with task homogenous windows
    print('++ INFO: Starting the evaluation phase....')
    print(aux.head(10))
    print(' +       Emb Size Pre-Eval = %s' % str(aux.shape))
    if type(aux.index) is pd.MultiIndex:
       print(' +       Index is MultiIndex Type')
       aux = aux.drop('XXXX', level='Window Name')
    else:
       print(' +       Index is Simple Index Type')
       aux = aux.drop('XXXX',axis=0)
    print(' +       Emb Size Post-Eval = %s' % str(aux.shape))
    print(' +       Looping through dimensions....')
    
    # Create Empty Dataframe that will hold all computed SI values
    # ============================================================
    df = pd.DataFrame(index=['SI_Subject','SI_Window Name'], columns=np.arange(2,m+1))
    df.columns.name = 'm'
    
    for m_max in np.arange(2,m+1):
          sel_dims = ['UMAP'+str(i+1).zfill(3) for i in np.arange(m_max)]
          print(' +      [%d/%d]  Dimensions = %s' % (m_max,m,str(sel_dims)))
          si_sbj  = silhouette_score(aux[sel_dims], aux.reset_index()['Subject'], n_jobs=-1)
          si_task = silhouette_score(aux[sel_dims], aux.reset_index()['Window Name'], n_jobs=-1)
          print(' +               SIsbj = %.2f | SItask= %.2f' % (si_sbj,si_task))
          df.loc['SI_Subject',m_max]     = si_sbj 
          df.loc['SI_Window Name',m_max] = si_task 
    print('++ INFO: Evaluation completed...')
    print(df)
    print('++ INFO: Saving SI values to disk... [%s]' % si_path)
    df.to_pickle(si_path)
    
###    umap_dims = [c for c in aux.columns if 'UMAP0' in c]
###    print('++ INFO: Computing SIsbj...')
###    si_sbj  = silhouette_score(aux[umap_dims], aux.reset_index()['Subject'], n_jobs=-1)
###    print(' +       SIsbj = %.2f' % si_sbj)
###    print('++ INFO: Computing SItask...')
###    si_task = silhouette_score(aux[umap_dims], aux.reset_index()['Window Name'], n_jobs=-1)
###    print(' +       SItask = %.2f' % si_task)
###    
###    # Writing SI results to disk
###    df = pd.Series(index=['SI_Subject','SI_Window Name'], dtype=float)
###    df['SI_Subject'] = si_sbj
###    df['SI_Window Name'] = si_task
###    print('++ INFO: Saving SI values to disk... [%s]' % si_path)
###    df.to_pickle(si_path)

def main():
    parser=argparse.ArgumentParser(description="Create a Laplacian Eigenmap embedding given a tvFC matrix")
    parser.add_argument("-sbj_list",    help="List of scans",                   dest="sbj_list",    type=str,  required=True)
    parser.add_argument("-input_data",  help="Input data",                      dest="input_data",  type=str,   required=True)
    parser.add_argument("-norm_method", help="FC matrix normalization method",  dest="norm_method", type=str,   required=True)
    parser.add_argument("-dist",        help="Distance Metric",                 dest="dist",        type=str,   required=True)
    parser.add_argument("-knn",         help="Number of Neighbors",             dest="knn",         type=int,   required=True)
    parser.add_argument("-m",           help="Final Dimensionality",            dest="m",           type=int,   required=True)
    parser.add_argument("-alpha",       help="Learning Rate",                   dest="alpha",       type=float,   required=True)
    parser.add_argument("-mdist",       help="Minimum Distance",                dest="mdist",       type=float, required=True)
    parser.add_argument("-init_method", help="Initialization Method",           dest="init_method", type=str,   required=True)
    parser.add_argument("-drop_xxxx",   help="Drop Task Inhomogenous Windows?", dest="drop_xxxx",    type=str,  required=True)

    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()