import pandas as pd
def load_netcc(path,roi_names=None):
 sfc = pd.read_csv(path,sep='\t',comment='#', header=0)
 sfc = sfc.drop(0,axis=0)
 sfc.index = sfc.columns
 if roi_names is not None:
  sfc.index = roi_names
  sfc.columns = roi_names
 return sfc
