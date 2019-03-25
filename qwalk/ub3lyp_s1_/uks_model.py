from uks_up_model import ED_uks_up
from uks_dn_model import ED_uks_dn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ED_uks(save=False):
  df=pd.concat((ED_uks_up(),ED_uks_dn()),axis=0)
  sns.pairplot(df,vars=['E','n_2ppi','n_2pz','n_4s','n_3d'],hue='Sz')
  if(save): plt.savefig('analysis/uks_eigenvalues.pdf',bbox_inches='tight'); plt.close()
  else: plt.show(); plt.close()

  return df
