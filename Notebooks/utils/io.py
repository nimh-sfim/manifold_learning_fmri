import pandas as pd
import numpy as np
from .basics import PRJ_DIR
from .basics import le_dist_metrics, le_knns, le_ms
from .basics import umap_dist_metrics, umap_knns, umap_ms, umap_alphas, umap_inits
from .basics import tsne_dist_metrics, tsne_pps, tsne_ms, tsne_alphas, tsne_inits
from .basics import input_datas, norm_methods

from tqdm.notebook import tqdm_notebook
import os.path as osp

def read_afni_roidata(path, verb=True):
    roi_info = pd.read_csv(path, engine='python', skipinitialspace=True, sep=' ', skiprows=1, header=None, names=['Nv','Nnnv','Frac','X','ROI','ROI_Orig_ID']).drop('X',axis=1).set_index('ROI')
    roi_info['Hemisphere'] = [r.split('_')[1] for r in roi_info['ROI_Orig_ID'] ]
    roi_info['Network']    = [r.split('_')[2] for r in roi_info['ROI_Orig_ID'] ]
    roi_info['ROI']        = ['_'.join(r.split('_')[3:]) for r in roi_info['ROI_Orig_ID'] ]
    print("++ INFO [read_afni_roidat]: Number of ROIs = %d " % roi_info.shape[0])
    return roi_info

def read_netts(roi_path,TR_secs,roi_names=None):
    roi_ts = pd.read_csv(roi_path, sep='\t', header=None).T
    roi_ts.columns.name = 'ROI_Name'
    if roi_names is None:
        roi_ts.columns = ['ROI{r}'.format(r=str(i).zfill(3)) for i in np.arange(roi_ts.shape[1])]
    else:
        roi_ts.names = roi_names
    roi_ts.index   = pd.timedelta_range(start='0',periods=roi_ts.shape[0],freq='{tr}L'.format(tr=TR_secs*1000))
    return roi_ts

def load_LE_SI(sbj_list,check_availability=False, verbose=False, wls=45, wss=1.5, ms=le_ms,dist_metrics=le_dist_metrics,knns=le_knns,norm_methods=norm_methods,input_datas=input_datas):
    """Loads Silhouette Index results for all available scans reduced via Laplacian Eigenmaps
    
       Inputs:
       -------
       sbj_list: List of subjects. Add 'ALL' to also load group-level results
       check_availability: if True, nothing is read from disk. It only check if files exists. Useful to check if all jobs have completed successfully.
       verbose: True means show more info about progress
       wls: window length in seconds [Default = 45 secs]
       wss: window spte in seconds [Default = 1.5 secs]
       ms: list of low dimensions. [Default = les_ms]
       dist_metrics: list of distance functions [Default = le_dist_metrics]
       knns: list of neighhorhood sizes [Default = le_knns]
       norm_methods: list of normalization schemes [Default = norm_methods]
       input_datas: list of input data types [Default = input_datas]
       
       Returns:
       --------
       si_LE: dataframe with one row per scan and the following columns: 'Subject','Norm','Metric','Knn','m','Target','SI'
              for individual scans, target should only include Window Name
              for group-level, target should include both Subject and Window Name
    """
    si_LE = pd.DataFrame(columns=['Subject','Input Data','Norm','Metric','Knn','m','Target','SI'])
    num_missing_files = 0
    num_needed_files = 0
    for m in tqdm_notebook(ms, desc='Final Dimensions'):
        for dist in tqdm_notebook(dist_metrics, desc='Distance Metric',leave=False):
            for knn in knns:
                for norm_method in norm_methods:
                    for sbj in sbj_list:
                        for input_data in input_datas:
                           num_needed_files +=1
                           path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{norm_method}.SI.pkl'.format(norm_method=norm_method,sbj=sbj,
                                                                                                                                         wls=str(int(wls)).zfill(3), 
                                                                                                                                         wss=str(wss),
                                                                                                                                         dist=dist,
                                                                                                                                         knn=str(knn).zfill(4),
                                                                                                                                         m=str(m).zfill(4)))
                           if not osp.exists(path):
                              num_missing_files +=1
                              if verbose:
                                  print('-e "%s"' % path, end=' ')
                              continue
                           if not check_availability:
                              aux = pd.read_pickle(path)
                              label_ids = list(aux.index)
                              for label in label_ids:
                                 si_LE = si_LE.append({'Subject':sbj,'Input Data':input_data,
                                                       'Norm':norm_method,'Metric':dist,
                                                       'Knn':knn,'m':m,'Target':label.split('_')[1],'SI':aux[label]}, ignore_index=True)
                              
    print('++ INFO: Number of files missing: [%d/%d] files' % (num_missing_files,num_needed_files))
    return si_LE

def load_UMAP_SI(sbj_list,check_availability=False, verbose=False, wls=45, wss=1.5, 
                 input_datas=input_datas,norm_methods=norm_methods,dist_metrics=umap_dist_metrics,init_methods=umap_inits,
                 knns=umap_knns,ms=umap_ms, alphas=umap_alphas, min_dist=0.8):
    si_UMAP = pd.DataFrame(columns=['Subject','Input Data','Norm','Metric','Knn','m','Alpha','Init','MinDist','Target','SI'])
    num_missing_files = 0
    num_needed_files = 0
    for sbj in tqdm_notebook(sbj_list, desc='Subjects:'):
        for input_data in tqdm_notebook(input_datas, desc='Data Inputs:',leave=False):
            for norm_method in norm_methods:
                for dist in dist_metrics:
                    for init_method in init_methods:
                        for knn in knns:
                            for m in ms:
                                for alpha in alphas:
                                   num_needed_files += 1
                                   path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.SI.pkl'.format(norm_method=norm_method,init_method=init_method,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                                   if not osp.exists(path):
                                       num_missing_files = num_missing_files + 1
                                       if verbose:
                                           print('-e "%s"' % path, end=' ')
                                       continue
                                   if not check_availability:
                                       aux = pd.read_pickle(path)
                                       label_ids = list(aux.index)
                                       for label in label_ids:
                                           si_UMAP = si_UMAP.append({'Subject':sbj,'Input Data':input_data,
                                                                     'Norm':norm_method,'Metric':dist,
                                                                     'Knn':knn,'m':m,'Alpha':alpha,'Init':init_method,
                                                                     'MinDist':min_dist,
                                                                     'Target':label.split('_')[1],'SI':aux[label]}, ignore_index=True)
                                           
    print('++ INFO: Number of missing files = [%d/%d] files' % (num_missing_files,num_needed_files))
    return si_UMAP
   
def load_TSNE_SI(sbj_list,check_availability=False, verbose=False, wls=45, wss=1.5,
                 input_datas=input_datas,norm_methods=norm_methods,dist_metrics=tsne_dist_metrics,
                 init_methods=tsne_inits,pps=tsne_pps, alphas=tsne_alphas, ms=tsne_ms, no_tqdm=False):
    si_TSNE = pd.DataFrame(columns=['Subject','Input Data','Norm','Metric','PP','m','Alpha','Init','Target','SI'])
    num_missing_files = 0
    num_needed_files = 0
    for sbj in tqdm_notebook(sbj_list, desc='Subjects:', disable=no_tqdm):
        for input_data in tqdm_notebook(input_datas, desc='Data Inputs:',leave=False,disable=no_tqdm):
            for norm_method in norm_methods:
                for dist in dist_metrics:
                    for pp in pps:
                        for m in ms:
                            for alpha in alphas:
                                for init_method in init_methods:
                                   num_needed_files += 1
                                   path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'TSNE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{lr}_{init_method}.{nm}.SI.pkl'.format(sbj=sbj,
                                                                                                                                                   nm = norm_method,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   init_method=init_method,
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   lr=str(alpha)))
                                   if not osp.exists(path):
                                       num_missing_files = num_missing_files + 1
                                       if verbose:
                                           print('-e "%s"' % path, end=' ')
                                       continue
                                   if not check_availability:
                                       aux = pd.read_pickle(path)
                                       label_ids = list(aux.index)
                                       for label in label_ids:
                                           si_TSNE = si_TSNE.append({'Subject':sbj,'Input Data':input_data,
                                                                     'Norm':norm_method,'Metric':dist,
                                                                     'PP':pp,'m':m,'Alpha':alpha,'Init':init_method,
                                                                     'Target':label.split('_')[1],'SI':aux[label]}, ignore_index=True)
                                           
    print('++ INFO: Number of missing files = [%d/%d] files' % (num_missing_files,num_needed_files))
    return si_TSNE
   

def load_single_le(sbj,input_data,scenario,dist,knn,m,wls=45,wss=1.5, drop_xxxx=True):
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{scenario}.pkl'.format(sbj=sbj,
                                                                                                                                         scenario=scenario,
                                                                                                                                         wls=str(int(wls)).zfill(3), 
                                                                                                                                         wss=str(wss),
                                                                                                                                         dist=dist,
                                                                                                                                         knn=str(knn).zfill(4),
                                                                                                                                         m=str(m).zfill(4)))
    if osp.exists(path):
        aux = pd.read_pickle(path)
        if drop_xxxx:
            if type(aux.index) is pd.MultiIndex:
                aux = aux.drop('XXXX', level='Window Name')
            else:
                aux = aux.drop('XXXX',axis=0)
        return aux
    else:
        return None
     
def load_LE_embeddings(sbj_list,wls=45,wss=1.5,dist_metrics=le_dist_metrics,knns=le_knns,ms=le_ms, norm_methods=norm_methods, input_datas=input_datas,check_availability=False, verbose=True, drop_xxxx=True):
    embs = {}
    num_missing_files = 0
    num_needed_files = 0
    for m in tqdm_notebook(ms, desc='Final Dimensions'):
        for dist in tqdm_notebook(dist_metrics, desc='Distance Metric',leave=False):
            for knn in knns:
                for norm_method in norm_methods:
                    for sbj in sbj_list:
                        for input_data in input_datas:
                            num_needed_files += 1
                            path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'LE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{norm_method}.pkl'.format(sbj=sbj,norm_method=norm_method,
                                                                                                                                         wls=str(int(wls)).zfill(3), 
                                                                                                                                         wss=str(wss),
                                                                                                                                         dist=dist,
                                                                                                                                         knn=str(knn).zfill(4),
                                                                                                                                         m=str(m).zfill(4)))
                            if not osp.exists(path):
                                num_missing_files += 1
                                if verbose:
                                   print('-e "%s"' % path, end='\n')
                                continue
                            if not check_availability:
                                aux = pd.read_pickle(path)
                                if type(aux.index) is pd.MultiIndex:
                                   aux = aux.drop('XXXX', level='Window Name')
                                else:
                                   aux = aux.drop('XXXX',axis=0)
                                embs[sbj,norm_method,dist,knn,m] = aux
    print('++ INFO: Number of missing files [%d/%d]' % (num_missing_files, num_needed_files))
    return embs
   
def load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,m,wls=45,wss=1.5, drop_xxxx=True):
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,init_method=init_method,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
    
    if osp.exists(path):
        aux = pd.read_pickle(path)
        if drop_xxxx:
            if type(aux.index) is pd.MultiIndex:
                aux = aux.drop('XXXX', level='Window Name')
            else:
                aux = aux.drop('XXXX',axis=0)
        return aux
    else:
        return None

def load_single_tsne(sbj,input_data,scenario,dist,pp,alpha,init_method,m,wls=45,wss=1.5, drop_xxxx=True):
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'TSNE',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.TSNE_{dist}_pp{pp}_m{m}_a{alpha}_{init_method}.{scenario}.pkl'.format(scenario=scenario,init_method=init_method,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   pp=str(pp).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   alpha=str(alpha)))
    
    if osp.exists(path):
        aux = pd.read_pickle(path)
        if drop_xxxx:
            if type(aux.index) is pd.MultiIndex:
                aux = aux.drop('XXXX', level='Window Name')
            else:
                aux = aux.drop('XXXX',axis=0)
        return aux
    else:
        print("++ WARNING: Missing File [%s]" % path)
        return None
     
def load_UMAP_embeddings(sbj_list=['ALL'],check_availability=False, verbose=False, wls=45, wss=1.5, ms=umap_ms,dist_metrics=umap_dist_metrics,knns=umap_knns, alphas=umap_alphas, norm_methods=norm_methods, init_methods=umap_inits, min_dist=0.8, input_datas=input_datas,drop_xxxx=True):
    embs = {}
    num_missing_files = 0
    num_needed_files = 0
    for m in tqdm_notebook(ms, desc='Final Dimensions'):
        for dist in tqdm_notebook(dist_metrics, desc='Distance Metric',leave=False):
            for knn in knns:
                for alpha in alphas:
                    for norm_method in norm_methods:
                        for sbj in sbj_list:
                            for init_method in init_methods:
                                for input_data in input_datas:
                                    num_needed_files += 1
                                    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,'UMAP',input_data,'{sbj}_Craddock_0200.WL{wls}s.WS{wss}s.UMAP_{dist}_k{knn}_m{m}_md{min_dist}_a{alpha}_{init_method}.{norm_method}.pkl'.format(norm_method=norm_method,init_method=init_method,sbj=sbj,
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss),
                                                                                                                                                   dist=dist,
                                                                                                                                                   knn=str(knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   min_dist=str(min_dist),
                                                                                                                                                   alpha=str(alpha)))
                                    if not osp.exists(path):
                                        num_missing_files = num_missing_files + 1
                                        if verbose:
                                            print('-e "%s"' % path, end='\n')
                                        continue
                                    if not check_availability:
                                        aux = pd.read_pickle(path)
                                        if drop_xxxx:
                                            if type(aux.index) is pd.MultiIndex:
                                                aux = aux.drop('XXXX', level='Window Name')
                                            else:
                                                aux = aux.drop('XXXX',axis=0)
                                        embs[sbj,norm_method,dist,knn,alpha,init_method,min_dist,m] = aux
    print('++ INFO: Number of missing files = [%d/%d] files' % (num_missing_files,num_needed_files))
    return embs