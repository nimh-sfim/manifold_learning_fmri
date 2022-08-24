PRJ_DIR = '/data/SFIMJGC_HCP7T/manifold_learning_fmri'

task_cmap      = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}
task_cmap_caps = {'REST': 'gray', 'BACK': 'blue',   'VIDE': '#F4D03F',  'MATH': 'green', 'XXXX': 'black'}

PNAS2015_folder = '/data/SFIMJGC/PRJ_CognitiveStateDetection01/'
PNAS2015_subject_list = ['SBJ06','SBJ07','SBJ08','SBJ09','SBJ11','SBJ12','SBJ13','SBJ16','SBJ17','SBJ20','SBJ21','SBJ22','SBJ23','SBJ24','SBJ25','SBJ26']
import pandas as pd

def load_representative_tvFC_data():
    print('++ INFO: Loading the tvFC dataset.....')
    X_df = pd.read_csv('../Resources/Figure03/swcZ_sbj06_ctask001_nroi0200_wl030_ws001.csv.gz', index_col=[0,1])
    # Becuase pandas does not like duplicate column names, it automatically adds .1, .2, etc to the names. We delete those next
    X_df.columns = X_df.columns.str.split('.').str[0]
    return X_df