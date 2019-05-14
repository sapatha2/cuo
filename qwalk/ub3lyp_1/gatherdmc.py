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
    f=basename+'/gsw'+str(np.round(gsw,2))+'_'+str(j)+'.dmc.gosling.json' 

    statinfo=os.stat(f)
    if(statinfo.st_size>0):
      print(f)
      data=json.load(open(f,'r'))
      obdm,__=get_qwalk_dm(data['properties']['tbdm_basis1'])
      obdm2,__,tbdm,__=get_qwalk_dm(data['properties']['tbdm_basis2'])
      energy=data['properties']['total_energy']['value'][0]*27.2114
      energy_err=data['properties']['total_energy']['error'][0]*27.2114

      print(obdm.shape,obdm2.shape,tbdm.shape)

      #MO ordering
      #1-body
      orb1=[0,1,2,3,4,5,6,7,8,9,1,2,3,8,3]
      orb2=[0,1,2,3,4,5,6,7,8,9,6,7,8,9,9]      
      mo=sum_onebody(obdm,orb1,orb2)
      mo_labels=['mo_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]
  
      orb1=[0,1,2,3,4,5,6,7,8,0,0,3,2,4]
      orb2=[0,1,2,3,4,5,6,7,8,3,8,8,7,6]
      iao=sum_onebody(obdm2,orb1,orb2)
      iao_labels=['iao_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]

      #2-body
      orb1=[0,1,2,3,4,5,6,7,8]
      u=sum_U(tbdm,orb1)
      u_labels=['u'+str(orb1[i]) for i in range(len(orb1))]

      #    |-Jsd----| |-Jdd-------------| |Jsp| |Jpp| |-Jdp-----------------------|
      orb1=[0,0,0,0,0,1,1,1,1,2,2,2,3,3,4,0,0,0,6,6,7,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
      orb2=[1,2,3,4,5,2,3,4,5,3,4,5,4,5,5,6,7,8,7,8,8,6,7,8,6,7,8,6,7,8,6,7,8,6,7,8]
      j=sum_J(tbdm,orb1,orb2)
      j_labels=['j_'+str(orb1[i])+'_'+str(orb2[i]) for i in range(len(orb1))]

      dat=np.array([energy,energy_err]+list(mo)+list(iao)+list(u)+list(j))
      d=pd.DataFrame(dat[:,np.newaxis].T,columns=['energy','energy_err']+mo_labels+iao_labels+u_labels+j_labels)
      d=d.astype('double')
      if(df is None): df=d
      else: df=pd.concat((df,d),axis=0)      
    else: print(f+' does not exist')
  fout=basename+'/dmc_gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns 
if __name__=='__main__':
  for basestate in np.arange(13):
    for gsw in [1.0]:
      N=10
      if(gsw==1.0): N=1
      gather_all(N,gsw,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
