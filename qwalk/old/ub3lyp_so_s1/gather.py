import json 
import pandas as pd
import numpy as np 
import sys
sys.path.append('../../../downfolding/')
from shivesh_downfold_tools import get_qwalk_dm, sum_onebody, sum_J, sum_U, sum_V
import os 

full_labels=np.array(["3s","3pz","3py","3px","2pz","3dz2","3dxz","3dyz"])
def gather_all(N,gsw,basename):
  ''' 
  Gathers all your data and stores into 
  '''
  df=None
  for j in range(1,N+1):
    f=basename+'/gsw'+str(np.round(gsw,2))+'_'+str(j)+'.vmc.gosling.json' 

    statinfo=os.stat(f)
    if(statinfo.st_size>0):
      print(f)
      data=json.load(open(f,'r'))
      obdm,__=get_qwalk_dm(data['properties']['tbdm_basis1'])
      __,__,tbdm,__=get_qwalk_dm(data['properties']['tbdm_basis2'])
      energy=data['properties']['total_energy']['value'][0]*27.2114
      energy_err=data['properties']['total_energy']['error'][0]*27.2114

      orb1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,5, 6 ,7 ,7 ,12]
      orb2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,10,11,12,13,13]
      one_body=sum_onebody(obdm,orb1,orb2)
      one_labels=['t_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]

      sigU=0.5*(data['properties']['tbdm_basis2']['tbdm']['updown'][0]+\
      data['properties']['tbdm_basis2']['tbdm']['downup'][0])

      dat=np.array([energy,energy_err,sigU]+list(one_body))
      d=pd.DataFrame(dat[:,np.newaxis].T,columns=['energy','energy_err','sigU']+one_labels)
      d=d.astype('double')
      d['gsw']=gsw
      if(j>N): d['gsw']=0
      if(df is None): df=d
      else: df=pd.concat((df,d),axis=0)      
    else: print(f+' does not exist')
  fout=basename+'/gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 
if __name__=='__main__':
  for basestate in np.arange(10):
    for gsw in np.arange(0.1,1.1,0.1): 
      if(gsw==1.0): N=1
      else: N=10
      gather_all(N,gsw,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))