#Analyze data 
import sys
#sys.path.append('/Users/shiveshpathak/Box Sync/Research/Work/si_model_fitting')
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analyze_jsonlog import gather_json_df
from scipy.stats import probplot

#cutoff=[0.0,0.25,0.50,0.75,1.0,1.5,2.0,2.5,3.0]
cutoff=[0.855,1.185,1.275,1.35]
vals=[]
errs=[]
for cut in cutoff:
  fname='Cuvtz0_B3LYP_s3_g0.1_c'+str(cut)+'.vmc.json'
  d=gather_json_df(fname)
  print(d.shape)
  print(d.mean()['dpwf_1'])
  print(d.std()['dpwf_1']/np.sqrt(d.shape[0]))

  res = probplot(d['dpwf_1'], plot=plt)
  plt.show()

  vals.append(d.mean()['dpwf_1'])
  errs.append(d.std()['dpwf_1']/np.sqrt(d.shape[0]))

plt.errorbar(cutoff,vals,yerr=errs,fmt='o')
plt.show()
