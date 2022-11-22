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

# +
import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
import seaborn as sns
import matplotlib.pyplot as plt
from utils.basics import PRJ_DIR
from utils.basics import task_cmap_caps
from utils.classification import scan_level_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

from utils.random import seed_value

from sklearn.preprocessing import MinMaxScaler

import hvplot.pandas

import pickle
# -

import sklearn
print(sklearn.__version__)

# Input is SWC
input_path   = '/data/SFIMJGC_HCP7T/manifold_learning_fmri/Data_Interim/PNAS2015/SBJ06/Original/SBJ06_Craddock_0200.WL045s.WS1.5s.tvFC.Z.asis.pkl'
feature_list =  ['All_Connections']

# Input is scan-level embedding
sbj, input_data, norm_method, emb_tech, dist,m = 'SBJ06','Original','asis','LE','correlation',30
input_path = osp.join(PRJ_DIR,'Data_Interim','PNAS2015',sbj,emb_tech,input_data,'{sbj}_Craddock_0200.WL045s.WS1.5s.{et}_{dist}_k0060_m{m}.{nm}.pkl'.format(sbj=sbj, et=emb_tech,dist=dist,m=str(m).zfill(4),nm=norm_method))
m = 30
features = ','.join(['LE'+str(i+1).zfill(3) for i in range(m)])
feature_list = features.split(',')

clf        = 'LR'
n_jobs     = 8
pid        = 'Window Name'
output_path = 'test.pkl'

# Read Input
# ===========
input_matrix = pd.read_pickle(input_path)
print(" + input_matrix.shape: %s" % str(input_matrix.shape))

# Curate Input
# ============
if 'All_Connections' in feature_list:
    print('++ INFO: Input is original SWC matrix --> Rearanging dataframe')
    input_matrix            = input_matrix.T
    input_matrix.index.name = 'Window Name'
    feature_list          = input_matrix.columns

# Drop XXXX windows from the input
# ================================
if type(input_matrix.index) is pd.MultiIndex:
    try:
        input_matrix = input_matrix.drop('XXXX',level='Window Name').copy()
    except:
        input_matrix = input_matrix.copy()
        print('++ WARNING: Dataframe does not contain XXXX entries. This should only be the case for group-level Procrustes data')
else:
    try:
        input_matrix = input_matrix.drop('XXXX').copy()
    except:
        input_matrix = input_matrix.copy()
        print('++ WARNING: Dataframe does not contain XXXX entries. This should only be the case for group-level Procrustes data')
print('++ INFO: Final Embedding DataFrame Size = %s' % str(input_matrix.shape))

# Check Requested Label set is available
# ======================================
num_label_sets = input_matrix.index.nlevels
label_sets = list(input_matrix.index.names)
print('++ INFO: # Available Label Sets = %d sets | Set Names: [%s]' % (num_label_sets, label_sets))
if pid not in label_sets:
    print('++ ERROR: Requested label set is not available. Program will end.')
    #return

# Extract Input Features for Classification
# =========================================
X = input_matrix[feature_list].values
print('++ INFO: Features extracted from dataframe. Final shape of feature array is : %s' % str(X.shape))

# Extract Categorical Labels for all available class problems
# ===========================================================
print('++ INFO: Extracting labels for all possible classification problems')
y_cat    = {cp:list(input_matrix.index.get_level_values(cp)) for cp in label_sets}

# Create Categorical -> Numerical Encoders and apply them to the data
# ===================================================================
print('++ INFO: Converting Categorical Labels to Numeric Labels')
lab_encs = {cp:LabelEncoder().fit(y_cat[cp]) for cp in label_sets}
y        = {cp:lab_encs[cp].transform(y_cat[cp]) for cp in label_sets}

# Create Classifier Object
# ========================
print('++ INFO: Creating Classifier Object [Selection = %s]' % clf)
if clf == 'SVM':
    clf_obj  = svm.SVC(kernel='linear', C=1, random_state=seed_value)
elif clf == 'LR':
    clf_obj  = LogisticRegression(random_state=seed_value,solver='liblinear', penalty='l1')
else:
    clf_obj = None
    print('++ ERROR: Selected Classifier [%s] is not available. Program will end.')
    #return
print(' +       --> %s' % str(clf_obj))

# Create Pipeline
# ===============
print('++ INFO: Creating Pipeline: Scaler + Classifier')
clf_pipeline = make_pipeline(MinMaxScaler(),clf_obj)
print('         --> %s' % str(clf_pipeline))

# Working with scan-level data and predicting task
# ================================================
if (pid == 'Window Name') & (X.shape[0] == 729):
    print('++ INFO: Working on Classification Problem [Window Name]')
    scan_level_cv =  scan_level_split()
    print('++ INFO: Running cross-validation...')
    cv_obj = cross_validate(clf_pipeline, X, y['Window Name'], cv=scan_level_cv, scoring=['f1_weighted'], return_train_score=True, return_estimator=True, n_jobs=n_jobs)
    print("++ INFO: Scoring --> %0.2f accuracy with a standard deviation of %0.2f" % (cv_obj['test_f1_weighted'].mean(), cv_obj['test_f1_weighted'].std()))

# Save everything to disk
# =======================
print('++ INFO: Saving results to disk...')
objects_to_save = {'input_path':input_path,
                   'input_matrix':input_matrix,
                   'feature_list':feature_list,
                   'clf':clf,
                   'pid':pid,
                   'X':X,
                   'y':y,
                   'y_cat':y_cat,
                   'cv_obj':cv_obj}
with open(output_path, "wb") as f:
    pickle.dump(objects_to_save, f)
print(' +      File created: %s' % output_path)



# ***







scores['test_f1_weighted']

print(classification_report(y_cat['Window Name'][365:729],lab_encs['Window Name'].inverse_transform(scores['estimator'][0].predict(X[365:729,:]))))

mm = MinMaxScaler()

input_data_mm = pd.DataFrame(mm.fit_transform(input_data),columns=input_data.columns, index=input_data.index)

plt.figure(figsize=(10,5))
plt.plot(input_data[('ROI077', 'ROI070')].reset_index(drop=True))
plt.plot(input_data[('ROI066', 'ROI039')].reset_index(drop=True))

plt.figure(figsize=(10,5))
plt.plot(input_data_mm[('ROI077', 'ROI070')].reset_index(drop=True))
plt.plot(input_data_mm[('ROI066', 'ROI039')].reset_index(drop=True))







df = pd.DataFrame(columns=['Estimator','Class','Dimension','Coef'])
for s in range(4):
    aux_class = list(lab_encs['Window Name'].inverse_transform(scores['estimator'][s]['logisticregression'].classes_))
    aux_coef  = scores['estimator'][s]['logisticregression'].coef_
    for ci,c in enumerate(aux_class):
        for i in range(aux_coef.shape[1]):
            df = df.append({'Estimator':s,'Class':c,'Dimension':i,'Coef':aux_coef[ci][i]},ignore_index=True)

input_data.columns[top_100]

df_summary = df.groupby(by=['Class','Dimension']).mean()
df_summary.reset_index(inplace=True)
df_summary = df_summary.pivot(index='Class',columns='Dimension',values='Coef').T
df_summary

fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.set(font_scale=1.5, style='whitegrid')
df_summary.plot(kind='bar',stacked=True, legend=None, color=task_cmap_caps, ax=ax)
ax.set_ylabel('Classifier Coefficients')

top_100 = list(df_summary.abs().sum(axis=1).sort_values(ascending=False).index[0:150])

a = df_summary.loc[top_100]

fig, ax = plt.subplots(1,1,figsize=(30,5))
sns.set(font_scale=1.5, style='whitegrid')
a.plot(kind='bar',stacked=True, legend=None, color=task_cmap_caps, ax=ax)
ax.set_ylabel('Classifier Coefficients')


