#Analyze data 
import numpy as np 
import sys
#sys.path.append('/Users/shiveshpathak/Box Sync/Research/Work/si_model_fitting')
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
from analyze_jsonlog import compute_and_save,gather_json_df,bootstrap

fnames=["Cuvtz0_B3LYP_s"+str(i)+"_g0."+str(j)+".vmc.json" for i in range(1,10) \
for j in range(1,10)]

for fname in fnames:
  df=gather_json_df(fname)

  #old way 
  rsdf=bootstrap(df,100)
  print(rsdf)
  print(rsdf.mean())
  print(rsdf.std())

  #Calculate values
  for i in range(10):
    df['dpenergy_'+str(i)]=df['dpwf_'+str(i)]*(df['energy'])
    df['de_'+str(i)]=df['dpwf_'+str(i)]*(df['energy']-df['energy'].mean()) #new way

  print(df.mean())
  print(df.std()/np.sqrt(800))

  break
