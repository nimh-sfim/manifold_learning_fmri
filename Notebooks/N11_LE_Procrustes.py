import argparse
import numpy as np
import pandas as pd
from utils.random         import seed_value
from utils.basics         import PRJ_DIR
from utils.procrustes     import procrustes_scan_le_embs
from utils.io             import load_LE_SI
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
    if args.drop_xxxx == "True":
       drop_xxxx = True
    elif args.drop_xxxx == "False":
       drop_xxxx = False
    else:
       print('++ ERROR: drop_xxxx has an invalid value. Program will end.')
       return
       
    emb_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','LE',input_data,
                                'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                   wss=str(wss),
                                                                                                                   dist=dist,
                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                   nm=norm_method,
                                                                                                                   m=str(m).zfill(4)))
    
    si_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','LE',input_data,
                               'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.SI.pkl'.format(wls=str(int(wls)).zfill(3), 
                                                                                                                     wss=str(wss),
                                                                                                                     dist=dist,
                                                                                                                     knn=str(knn).zfill(4),
                                                                                                                     nm=norm_method,
                                                                                                                     m=str(m).zfill(4)))
    print(' ')
    print('++ INFO: Run information')
    print(' +       Seed Value              :', seed_value)
    print(' +       Input Data              :', input_data)
    print(' +       Scan List               :', str(sbj_list))
    print(' +       Normalization Method    :', norm_method)
    print(' +       Distance Function       :', dist)
    print(' +       Knn                     :', knn)
    print(' +       Final Dimensionality    :', m)
    print(' +       Drop XXXX               :', drop_xxxx)
    print(' +       Procrusted Emb Out Path :', emb_path)
    print(' +       SI Out Path             :', si_path)
    print(' ')
    
    # Load SI for the embeddings to be aligned
    print('++ INFO: Loading Scan-level SI for all participanting scans...')
    si_LE = load_LE_SI(sbj_list=sbj_list,check_availability=False, verbose=False, wls=wls, wss=wss, 
                           input_datas=[input_data], norm_methods=[norm_method], dist_metrics=[dist],
                           knns=[knn], ms=[m], no_tqdm=True)
    print(' + si_LE.shape = %s' % str(si_LE.shape))
    si_LE = si_LE.set_index(['Subject','Input Data','Norm','Metric','Knn','m','Target']).sort_index()
    print(si_LE)
    
    # Compute Procrustes Transformation
    print('++ INFO: Computing Procrustes Transformation....')
    aux = procrustes_scan_le_embs(sbj_list,si_LE,input_data,norm_method,dist,knn,m,drop_xxxx=drop_xxxx)
    aux = aux.set_index(['Subject','Window Name'])
    aux.columns.name = 'LE dimensions'
    print('++ INFO: Saving Group Level Embedding to disk...[%s]' % emb_path)
    aux.to_pickle(emb_path)
    
    # Evaluate the Quality of the Embedding with the SI index
    # Evaluation is always only with task homogenous windows
    print('++ INFO: Starting the evaluation phase....')
    print(' +       Emb Size Pre-Eval = %s' % str(aux.shape))
    if type(aux.index) is pd.MultiIndex:
       aux = aux.drop('XXXX', level='Window Name')
    else:
       aux = aux.drop('XXXX',axis=0)
    print(' +       Emb Size Post-Eval = %s' % str(aux.shape))
    print(' +       Looping through dimensions....')
    # Create Empty Dataframe that will hold all computed SI values
    # ============================================================
    df = pd.DataFrame(index=['SI_Subject','SI_Window Name'], columns=np.arange(2,m+1))
    df.columns.name = 'm'
    
    for m_max in np.arange(2,m+1):
          le_dims = ['LE'+str(i+1).zfill(3) for i in np.arange(m_max)]
          print(' +      [%d/%d]  Dimensions = %s' % (m_max,m,str(le_dims)))
          si_sbj  = silhouette_score(aux[le_dims], aux.reset_index()['Subject'], n_jobs=-1)
          si_task = silhouette_score(aux[le_dims], aux.reset_index()['Window Name'], n_jobs=-1)
          print(' +               SIsbj = %.2f | SItask= %.2f' % (si_sbj,si_task))
          df.loc['SI_Subject',m_max]     = si_sbj 
          df.loc['SI_Window Name',m_max] = si_task 
    print('++ INFO: Evaluation completed...')
    print(df)
    print('++ INFO: Saving SI values to disk... [%s]' % si_path)
    df.to_pickle(si_path)
def main():
    parser=argparse.ArgumentParser(description="Apply Procrustes Transformation to Scan-level LE Embeddings")
    parser.add_argument("-sbj_list",    help="List of scans",                   dest="sbj_list",    type=str,  required=True)
    parser.add_argument("-input_data",  help="Input data",                      dest="input_data",  type=str,  required=True)
    parser.add_argument("-norm_method", help="FC matrix normalization method",  dest="norm_method", type=str,  required=True)
    parser.add_argument("-dist",        help="Distance Metric",                 dest="dist",        type=str,  required=True)
    parser.add_argument("-knn",         help="Neighborhood Size",               dest="knn",         type=int,  required=True)
    parser.add_argument("-m",           help="Final Dimensionality",            dest="m",           type=int,  required=True)
    parser.add_argument("-drop_xxxx",   help="Drop Task Inhomogenous Windows?", dest="drop_xxxx",   type=str,  required=True)

    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
