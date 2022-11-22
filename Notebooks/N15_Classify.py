import argparse
import pandas as pd
import numpy as np
import pickle

from utils.random         import seed_value
from utils.classification import scan_level_split, group_level_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

def run(args):
    input_path      = args.input_path
    output_path     = args.output_path
    feature_list    = args.features.split(',')
    clf             = args.clf
    pid             = args.pid
    n_jobs          = args.n_jobs
    print(' ')
    print('++ INFO: Run information')
    print(' +       Input path       :', input_path)
    print(' +       Ouput path       :', output_path)
    print(' +       Feature List     :', str(feature_list))
    print(' +       Classifier       :', clf)
    print(' +       Target Labels    :', pid)
    print(' +       # Jobs           :', n_jobs)
    print(' +       Random Seed      :', seed_value)
    print(' ')

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
        return

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
    if clf == 'svc':
        clf_obj  = svm.SVC(kernel='linear', C=1, random_state=seed_value)
    elif clf == 'logisticregression':
        clf_obj  = LogisticRegression(random_state=seed_value,solver='liblinear', penalty='l1')
    else:
        clf_obj = None
        print('++ ERROR: Selected Classifier [%s] is not available. Program will end.')
        return
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
    # Working with group-level data and predicting task
    # =================================================
    if (pid == 'Window Name') & (X.shape[0] == 14580):
        print('++ INFO: Working on Classification Problem [Window Name]')
        group_level_cv =  group_level_split()
        print('++ INFO: Running cross-validation...')
        cv_obj = cross_validate(clf_pipeline, X, y['Window Name'], cv=group_level_cv, scoring=['f1_weighted'], return_train_score=True, return_estimator=True, n_jobs=n_jobs)
        print("++ INFO: Scoring --> %0.2f accuracy with a standard deviation of %0.2f" % (cv_obj['test_f1_weighted'].mean(), cv_obj['test_f1_weighted'].std()))
    # Save Results from Cross-Validation to Disk
    # ==========================================
    print('++ INFO: Saving results to disk...')
    objects_to_save = {'input_path':input_path,
                       'input_matrix':input_matrix,
                       'clf':clf,
                       'pid':pid,
                       'feature_list':feature_list,
                       'X':X,
                       'y':y,
                       'y_cat':y_cat,
                       'lab_encs':lab_encs,
                       'cv_obj':cv_obj}
    with open(output_path, "wb") as f:
        pickle.dump(objects_to_save, f)
    print(' +      File created: %s' % output_path)
        
        
def main():
    parser=argparse.ArgumentParser(description="Classifier-based evaluation of embeddings")
    parser.add_argument("-input_path",   help="Input path",                            dest="input_path",  type=str,  required=True)
    parser.add_argument("-features",     help="Features to use (separated by commas)", dest="features",    type=str,  required=True)
    parser.add_argument("-clf",          help="Classifier to use",                     dest="clf",         type=str,  choices=['logisticregression','svc'], required=True)
    parser.add_argument("-pid",          help="Classification problem to solve",       dest="pid",         type=str,  choices=['Window Name','Subject'], required=True)
    parser.add_argument("-output_path",  help="Ouptut path",                           dest="output_path", type=str,  required=True)
    parser.add_argument("-n_jobs",       help="Number of Jobs",                        dest="n_jobs",      type=int,  required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
