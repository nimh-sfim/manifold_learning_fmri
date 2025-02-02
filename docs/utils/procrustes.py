import pandas as pd
import numpy as np
from .io import load_single_le, load_single_tsne, load_single_umap
from scipy.stats import zscore
from scipy.spatial import procrustes

def procrustes_scan_le_embs(sbj_list,si_LE,input_data,scenario,dist,knn,m,drop_xxxx):
    embs = {}
    # Load all embeddings
    for sbj in sbj_list:
        embs[sbj] = load_single_le(sbj,input_data,scenario,dist,knn,m, drop_xxxx=drop_xxxx)
    # Select best embedding (based on SI) to be used as reference
    si_sel_data = si_LE.loc[sbj_list].loc[:,input_data,scenario,dist,:,m]
    best        = si_sel_data.sort_values(by='SI', ascending=False).iloc[0].name[0]
    # Get a list with the names of all other scans that need to be transformed
    scans_to_transform = sbj_list.copy()
    scans_to_transform.remove(best)
    # Copy the embedding to use as reference into the ref variable
    ref = embs[best]
    # Create object that will contain all overlapped embeddings
    all_embs            = zscore(ref.copy()).reset_index()
    sub_col = list(np.repeat(best,all_embs.shape[0]))
    # Go one-by-one computing transformation and keeping it 
    for scan in scans_to_transform:
        aux          = embs[scan]
        _, aux_trf,_ = procrustes(ref,aux)
        aux_trf      = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
        all_embs     = all_embs.append(aux_trf).reset_index(drop=True)
        sub_col      = sub_col + list(np.repeat(scan,aux.shape[0]))
    all_embs['Subject'] = sub_col
    # Drop In-between windows if requested
    #if drop_xxxx:
    #    all_embs = all_embs.set_index('Window Name').drop('XXXX').reset_index()
    return all_embs
   
### def procrustes_scan_umap_embs(sbj_list,si_UMAP,input_data,scenario,dist,knn,alpha, init_method, min_dist,m,drop_xxxx):
###     embs = {}
###     # Load all embeddings
###     for sbj in sbj_list:
###         embs[sbj] = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,m, drop_xxxx=False)
###     # Select best embedding (based on SI) to be used as reference
###     si_sel_data = si_UMAP.loc[sbj_list].loc[:,input_data,scenario,dist,:,m,alpha,init_method,:]
###     best        = si_sel_data.sort_values(by='SI', ascending=False).iloc[0].name[0]
###     # Get a list with the names of all other scans that need to be transformed
###     scans_to_transform = sbj_list.copy()
###     scans_to_transform.remove(best)
###     # Copy the embedding to use as reference into the ref variable
###     ref = embs[best]
###     # Create object that will contain all overlapped embeddings
###     all_embs            = zscore(ref.copy()).reset_index()
###     sub_col = list(np.repeat(best,all_embs.shape[0]))
###     # Go one-by-one computing transformation and keeping it 
###     for scan in scans_to_transform:
###         aux          = embs[scan]
###         _, aux_trf,_ = procrustes(ref,aux)
###         aux_trf      = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
###         all_embs     = all_embs.append(aux_trf).reset_index(drop=True)
###         sub_col      = sub_col + list(np.repeat(scan,aux.shape[0]))
###     all_embs['Subject'] = sub_col
###     # Drop In-between windows if requested
###     if drop_xxxx:
###         all_embs = all_embs.set_index('Window Name').drop('XXXX').reset_index()
###     return all_embs
   
def procrustes_scan_tsne_embs(sbj_list,si_TSNE,input_data,norm_method,dist,pp,m,alpha, init_method, drop_xxxx):
    embs = {}
    bad_embs = []
    # Load all embeddings
    for sbj in sbj_list:
        embs[sbj] = load_single_tsne(sbj,input_data,norm_method,dist,pp,alpha,init_method,m,drop_xxxx=drop_xxxx)
        if embs[sbj] is None:
            print('++ ERROR: Could not load scan_level_embedding [%s,%s,%s,%s,%d,%.2f,%s,%d]' % (sbj,input_data,norm_method,dist,pp,alpha,init_method,m))
            return None 
        stdv      = embs[sbj].std().values
        if np.any(np.isclose(stdv,0,rtol=1e-4,atol=1e-4)):
            print('++ WARNING: Scan with cramped embedding [%s,%s,%s,%s,%d,%.2f,%s,%d]' %(sbj,input_data,norm_method,dist,pp,alpha,init_method,m))
            bad_embs.append(sbj)
    # Modify list of scans to show if needed
    scan_list = [sbj for sbj in sbj_list if sbj not in bad_embs]
    
    # Select best embedding (based on SI) to be used as reference
    si_sel_data = si_TSNE.loc[scan_list].loc[:,input_data,norm_method,dist,:,m,alpha,init_method]
    best        = si_sel_data.sort_values(by='SI', ascending=False).iloc[0].name[0]
    
    # Get a list with the names of all other scans that need to be transformed
    scans_to_transform = scan_list.copy()
    scans_to_transform.remove(best)
    
    # Copy the embedding to use as reference into the ref variable
    ref = embs[best]
    
    # Create object that will contain all overlapped embeddings
    all_embs            = zscore(ref.copy()).reset_index()
    sub_col = list(np.repeat(best,all_embs.shape[0]))
    
    # Go one-by-one computing transformation and keeping it 
    for scan in scans_to_transform:
        aux          = embs[scan]
        try:
            _, aux_trf,_ = procrustes(ref,aux)
            aux_trf      = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
            all_embs     = all_embs.append(aux_trf).reset_index(drop=True)
            sub_col      = sub_col + list(np.repeat(scan,aux.shape[0]))
        except:
            print('++ ERROR: There are enough controls not to enter here: [%s,%s,%s,%s,%d,%.2f,%s,%d]' %(sbj,input_data,norm_method,dist,pp,alpha,init_method,m))
    all_embs['Subject'] = sub_col
    # Drop In-between windows if requested
    if drop_xxxx:
        all_embs = all_embs.set_index('Window Name').drop('XXXX').reset_index()
    return all_embs
   
def procrustes_scan_umap_embs(sbj_list,si_UMAP,input_data,scenario,dist,knn,alpha, init_method, min_dist,m,drop_xxxx):
    embs = {}
    # Load all embeddings
    for sbj in sbj_list:
        embs[sbj] = load_single_umap(sbj,input_data,scenario,dist,knn,alpha,init_method,min_dist,m, drop_xxxx=False)
        if embs[sbj] is None:
            print('++ ERROR: Could not load scan_level_embedding [%s,%s,%s,%s,%d,%.2f,%s,%d]' % (sbj,input_data,norm_method,dist,pp,alpha,init_method,m))
            return None 
    # Select best embedding (based on SI) to be used as reference
    si_sel_data = si_UMAP.loc[sbj_list].loc[:,input_data,scenario,dist,:,m,alpha,init_method,:]
    best        = si_sel_data.sort_values(by='SI', ascending=False).iloc[0].name[0]
    # Get a list with the names of all other scans that need to be transformed
    scans_to_transform = sbj_list.copy()
    scans_to_transform.remove(best)
    # Copy the embedding to use as reference into the ref variable
    ref = embs[best]
    # Create object that will contain all overlapped embeddings
    all_embs            = zscore(ref.copy()).reset_index()
    sub_col = list(np.repeat(best,all_embs.shape[0]))
    # Go one-by-one computing transformation and keeping it 
    for scan in scans_to_transform:
        aux          = embs[scan]
        _, aux_trf,_ = procrustes(ref,aux)
        aux_trf      = pd.DataFrame(zscore(aux_trf),index=aux.index, columns=aux.columns).reset_index()
        all_embs     = all_embs.append(aux_trf).reset_index(drop=True)
        sub_col      = sub_col + list(np.repeat(scan,aux.shape[0]))
    all_embs['Subject'] = sub_col
    # Drop In-between windows if requested
    if drop_xxxx:
        all_embs = all_embs.set_index('Window Name').drop('XXXX').reset_index()
    return all_embs