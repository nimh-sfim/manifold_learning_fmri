import argparse
import pandas as pd
from sklearn.metrics import silhouette_score

def run(args):
    input_path   = args.input_path
    output_path  = args.output_path
    print('')
    print('++ INFO: Run information')
    print(' + Input Path  :', input_path)
    print(' + Output Path :', output_path)
    print('')
    
    # Read Embedding
    # ==============
    print('++ INFO: Loading Embedding and labels')
    emb = pd.read_pickle(input_path)
    print(' + Original Embedding DataFrame Size = %s' % str(emb.shape))
    # Temporary Fix for UMAP
    emb.index.name = 'Window Name'
    emb.to_pickle(input_path)
    # End of Temporary Fix for UMAP
   
    # Temporary fix for Procrustes / TSNE
    if ('TSNE' in input_path) & ('Procrustes' in input_path):
       emb = emb.set_index(['Subject','Window Name'])
       emb.columns.name = 'TSNE Dimensions'
    # Temporary fix for Procrustes / TSNE END 
    n_labels  = emb.index.nlevels
    label_ids = list(emb.index.names)
    print(' + Number of labels = %d' % n_labels)
    print(' + Label IDs = %s' % str(label_ids))

    # Remove XXXX windows
    # ===================
    if type(emb.index) is pd.MultiIndex:
       try:
          emb_pure = emb.drop('XXXX',level='Window Name').copy()
       except:
          emb_pure = emb.copy()
          print('++ WARNING: Dataframe does not contain XXXX entries. This should only be the case for group-level Procrustes data')
    else:
       try:
          emb_pure = emb.drop('XXXX').copy()
       except:
          emb_pure = emb.copy()
          print('++ WARNING: Dataframe does not contain XXXX entries. This should only be the case for group-level Procrustes data')
    print(' + Final Embedding DataFrame Size = %s' % str(emb_pure.shape))
    # Compute Silhouette Index
    # ========================
    df = pd.Series(index=['SI_'+labelID for labelID in label_ids], dtype=float)
    for labelID in label_ids:
        df['SI_'+labelID] = silhouette_score(emb_pure, emb_pure.index.get_level_values(labelID), n_jobs=-1)
    
    print(' + ', str(df))
    # Save to Disk
    # ============
    print(' ')
    print('++ INFO: Save to disk [%s]' % output_path)
    df.to_pickle(output_path)
def main():
    parser=argparse.ArgumentParser(description="Compute Silhouette Index")
    parser.add_argument("-input",   help="Path to embedding in a dataframe pickled object",  dest="input_path", type=str,  required=True)
    parser.add_argument("-output",  help="Path to output file",  dest="output_path", type=str,  required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)
    
if __name__ == "__main__":
    main()
