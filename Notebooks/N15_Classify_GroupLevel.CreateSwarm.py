# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: opentsne
#     language: python
#     name: opentsne
# ---

import pandas as pd
import numpy as np
import os
import os.path as osp
import getpass
from datetime import datetime
from tqdm.notebook import tqdm
from utils.basics import task_cmap_caps
from utils.basics import PNAS2015_subject_list, PNAS2015_folder, PNAS2015_roi_names_path, PNAS2015_win_names_paths, PRJ_DIR, input_datas, norm_methods

wls = 45
wss = 1.5
tr  = 1.5
win_names_path = PNAS2015_win_names_paths[(wls,wss)]
umap_min_dist    = 0.8
umap_init_method = 'spectral'
tsne_init_method = 'pca'

# Given that we will use SI as a way to check optimal embeddings, we first load the results for the three modalities

si_LE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_LE.pkl'))

si_UMAP = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_UMAP.pkl'))

si_TSNE = pd.read_pickle(osp.join(PRJ_DIR,'Dashboard','Data','si_TSNE.pkl'))

# ***
# # Group Level - Procrustes

for input_data in input_datas:
    path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','LE',input_data)
    if not osp.exists(path):
        print('++ INFO: Creating new folder [%s]' % path)
        os.makedirs(path)

_,_,le_best_nm, le_best_dist, le_best_knn, _, _ = si_LE.loc['Procrustes','Original',:,:,:,3,'Task'].sort_values(by='SI', ascending=False).iloc[0].name
print('++ INFO: Procrustes Group Level - Best Hyper parameters: [%s,%s,%d] --> %.2f' % (le_best_nm, le_best_dist, le_best_knn,si_LE.loc['Procrustes','Original',:,:,:,:,'Task'].sort_values(by='SI', ascending=False).iloc[0]))

# +
#user specific folders
#=====================
username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))  

swarm_path     = osp.join(swarm_folder,'N15_Classify_GroupLevel_Procrustes_LE.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'N15_Classify_GroupLevel_Procrustes_LE.logs')

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
print('++ INFO: Swarm File  : %s' % swarm_path)
print('++ INFO: Logs Folder : %s' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')

# Insert comment line with SWARM command
swarm_file.write('#swarm -J Clf_Group_LE_Procrustes -f {swarm_path} -b 20 -g 8 -t 8 --time 00:10:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')


for input_data in input_datas:
        for clf in 'logisticregression','svc':
            for m in [2,3,5,10,15,20,25,30]:
                features = ','.join(['LE'+str(i+1).zfill(3) for i in range(m)])
                input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','LE',input_data,'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.pkl'.format(nm=le_best_nm,dist=le_best_dist,knn=str(le_best_knn).zfill(4),
                                                                                                                                                   m=str(m).zfill(4),
                                                                                                                                                   wls=str(int(wls)).zfill(3), 
                                                                                                                                                   wss=str(wss)))
                output_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','LE',input_data,'Procrustes_Craddock_0200.WL{wls}s.WS{wss}s.LE_{dist}_k{knn}_m{m}.{nm}.clf_results.{clf}_WindowName.pkl'.format(nm=le_best_nm,dist=le_best_dist,knn=str(le_best_knn).zfill(4),
                                                                                                                                                   clf=clf,
                                                                                                                                                   wls=str(int(wls)).zfill(3),m=str(m).zfill(4),
                                                                                                                                                   wss=str(wss)))
                                        
                swarm_file.write("export input_path={input_path}  output_path={output_path} clf={clf} pid='Window Name' features='{features}' n_jobs=8; sh {scripts_dir}/N15_Classify.sh".format(
                       input_path = input_path, output_path=output_path, clf=clf, features=features,
                       scripts_dir=osp.join(PRJ_DIR,'Notebooks')))
                swarm_file.write('\n')
swarm_file.close()
# -
import xarray as xr
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# %%time
LE_F1     = pd.DataFrame(columns=['Input','Subject','m','Classifier','F1'])
LE_COEFFS = {}
for m in tqdm([2,3,5,10,15,20,25,30],desc='Max Dimensions'):
    for input_data in input_datas:
        for clf in ['logisticregression','svc']:
            LE_COEFFS[(m,input_data,clf)] = xr.DataArray(dims=['Subject','Dimension','Class'], coords={'Subject':['Procrustes'],'Dimension':['LE'+str(i+1).zfill(4) for i in range(m)],'Class':['BACK','MATH','REST','VIDE']})
            # Load Classification results
            # ===========================
            path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015','Procrustes','Classification','LE',input_data,'Procrustes_Craddock_0200.WL045s.WS1.5s.LE_correlation_k0075_m{m}.zscored.clf_results.{clf}_WindowName.pkl'.format(m=str(m).zfill(4),clf=clf))
            with open(path,'rb') as f:
                objects = pickle.load(f)
            locals().update(objects)
            # Gather overall F1 - score
            # =========================
            LE_F1 = LE_F1.append({'Subject':'Procrustes','Input':input_data,'m':m,'Classifier':clf,
                                      'F1':cv_obj['test_f1_weighted'].mean()}, ignore_index=True)
                
            # Coefficients
            # ============
            df_coeffs = pd.DataFrame(columns=['Split','Class','Dimension','Coef'])
            for split in range(2):
                aux_class = list(lab_encs['Window Name'].inverse_transform(cv_obj['estimator'][split][clf].classes_))
                aux_coef  = cv_obj['estimator'][split][clf].coef_
                for ci,c in enumerate(aux_class):
                    for i in range(aux_coef.shape[1]):
                        df_coeffs = df_coeffs.append({'Split':split,'Class':c,'Dimension':feature_list[i],'Coef':abs(aux_coef[ci][i])},ignore_index=True)
            df_coeffs = df_coeffs.groupby(by=['Class','Dimension']).mean()
            df_coeffs.reset_index(inplace=True)
            df_coeffs = df_coeffs.pivot(index='Class',columns='Dimension',values='Coef').T
            LE_COEFFS[(m,input_data,clf)].loc['Procrustes',:,:] = df_coeffs
            del df_coeffs

sns.set(font_scale=1.5, style='whitegrid')
fig, axs = plt.subplots(1,2,figsize=(20,5))
for i,clf in enumerate(['logisticregression','svc']):
    aux_df = LE_F1.set_index(['Classifier']).sort_index().loc[clf].sort_values(by='Input', ascending=False)
    g = sns.barplot(data=aux_df,y='F1',x='Input',hue='m', ax=axs[i], alpha=.9, palette=sns.color_palette('Blues',8), edgecolor='k')
    #g = sns.swarmplot(data=aux_df,y='F1',x='Input',hue='m', ax=axs[i])
    g.set_ylim(0,1)
    g.set_title(clf)
    g.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),ncol=4, fancybox=True, shadow=True)
    g.set_xlabel('Data Model')

sns.set(font_scale=1.5, style='whitegrid')
fig, ax = plt.subplots(1,1,figsize=(7,5))
aux_df = LE_F1[LE_F1['Input']=='Original'].set_index(['Classifier']).sort_index().loc['logisticregression'].sort_values(by='Input', ascending=False)
g = sns.barplot(data=aux_df,y='F1',x='m', ax=ax, alpha=.9, palette=sns.color_palette('Blues',8), edgecolor='k')
g.set_ylim(0,1)
#g.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),ncol=4, fancybox=True, shadow=True)
g.set_xlabel('Final Dimensionality [m]');
g.set_ylabel('Accuracy [F1]');
g.set_title('Logistic Regression')

sns.set(font_scale=1.5, style='whitegrid')
fig, ax = plt.subplots(1,1,figsize=(20,5))
df_summary = LE_COEFFS[(30,'Original','logisticregression')].mean(dim='Subject').to_dataframe(name='Coeffs')
df_summary = df_summary.groupby(by=['Class','Dimension']).mean()
df_summary.reset_index(inplace=True)
df_summary = df_summary.pivot(index='Class',columns='Dimension',values='Coeffs').T
df_summary.index = [str(i+1) for i in range(30)]
df_summary.index.name = 'Laplacian Eigenmap Dimension'
df_summary.plot(kind='bar',stacked=True, legend=None, color=task_cmap_caps, ax=ax)
ax.set_ylabel('Logistic Regression Coefficients')
#ax.set_ylim(0,20)
#ax.set_title('Data: %s' % input_data)

LE_COEFFS[(25,
  'Original',
  'logisticregression')]

sns.set(font_scale=1.5, style='whitegrid')
fig, axs = plt.subplots(1,3,figsize=(30,5))
for i, input_data in enumerate(input_datas):
    df_summary = LE_COEFFS[(25,input_data,'logisticregression')].mean(dim='Subject').to_dataframe(name='Coeffs')
    df_summary = df_summary.groupby(by=['Class','Dimension']).mean()
    df_summary.reset_index(inplace=True)
    df_summary = df_summary.pivot(index='Class',columns='Dimension',values='Coeffs').T
    df_summary.plot(kind='bar',stacked=True, legend=None, color=task_cmap_caps, ax=axs[i])
    axs[i].set_ylabel('Logistic Regression Coefficients')
    #axs[i].set_ylim(0,20)
    axs[i].set_title('Data: %s' % input_data)

LE_COEFFS


