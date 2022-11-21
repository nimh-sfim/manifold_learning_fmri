import argparse
import pandas as pd
import numpy as np
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
    print('++ INFO: Loading Embedding and labels: [%s]' % input_path)
    emb = pd.read_pickle(input_path)
    print(' +       emb.shape=%s' % str(emb.shape))
    n_wins,m = emb.shape
    print(' + Original Embedding DataFrame Size = %s' % str(emb.shape))
    print(' + Number of Dimensions = %d' % m)
    # Temporary Fix for UMAP
    emb.index.name = 'Window Name'
    emb.to_pickle(input_path)
    # End of Temporary Fix for UMAP
   
    # Temporary fix for Procrustes / TSNE
    if ('TSNE' in input_path) & ('Procrustes' in input_path):
       print('++ WARNING: Index correction for TSNE.........................................')
       emb = emb.set_index(['Subject','Window Name'])
       emb.columns.name = 'TSNE Dimensions'
    # Temporary fix for Procrustes / TSNE END 
    n_labels  = emb.index.nlevels
    label_ids = list(emb.index.names)
    print(' + Number of labels = %d' % n_labels)
    print(' + Label IDs = %s' % str(label_ids))

    # Remove XXXX windows
    # ===================
    print('++ INFO: Starting Embedding Shape       = %s' % str(emb.shape))
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
    print(' +       Final Embedding DataFrame Size = %s' % str(emb_pure.shape))
    
    # Create Empty Dataframe that will hold all computed SI values
    # ============================================================
    df = pd.DataFrame(index=['SI_'+labelID for labelID in label_ids],columns=np.arange(2,m+1))
    df.columns.name = 'm'
    # Compute SI for all possible m values up to the maximum available
    # ================================================================
    for m_max in np.arange(2,m+1):
          if ('LE' in input_path):
              sel_dims = ['LE'+str(i+1).zfill(3) for i in np.arange(m_max)]
          if ('TSNE' in input_path):
              sel_dims = ['TSNE'+str(i+1).zfill(3) for i in np.arange(m_max)]
          if ('UMAP' in input_path):
              sel_dims = ['UMAP'+str(i+1).zfill(3) for i in np.arange(m_max)]
          print(' +      [%d/%d]  Dimensions = %s' % (m_max,m,str(sel_dims)))
          for labelID in label_ids: 
               si_value  = silhouette_score(emb[sel_dims], emb.reset_index()[labelID], n_jobs=-1)
               print(' +               SI_%s = %.2f' % (labelID,si_value))
               df.loc['SI_'+labelID,m_max]     = si_value
    print('++ INFO: Evaluation completed...')
    print(df)
    
    # Save dataframe with all SI values to disk
    # =========================================
    print('++ INFO: Saving SI values to disk... [%s]' % output_path)
    df.to_pickle(output_path)
    
    # PRIOR CODE - ONLY ONE M # # Compute Silhouette Index
    # PRIOR CODE - ONLY ONE M # # ========================
    # PRIOR CODE - ONLY ONE M # df = pd.Series(index=['SI_'+labelID for labelID in label_ids], dtype=float)
    # PRIOR CODE - ONLY ONE M # for labelID in label_ids:
    # PRIOR CODE - ONLY ONE M #     df['SI_'+labelID] = silhouette_score(emb_pure, emb_pure.index.get_level_values(labelID), n_jobs=-1)
    # PRIOR CODE - ONLY ONE M # 
    # PRIOR CODE - ONLY ONE M # print(' + ', str(df))
def main():
    parser=argparse.ArgumentParser(description="Compute Silhouette Index")
    parser.add_argument("-input",   help="Path to embedding in a dataframe pickled object",  dest="input_path", type=str,  required=True)
    parser.add_argument("-output",  help="Path to output file",  dest="output_path", type=str,  required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)
    
if __name__ == "__main__":
    main()
