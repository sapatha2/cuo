import json 
import pandas as pd
import numpy as np 
import sys
sys.path.append('../../../downfolding/')
from shivesh_downfold_tools import get_qwalk_dm

def gather_all(detgen,N,Ndet,gsw,basename):
  ''' 
  Gathers all your data and stores into 
  '''
  df=None
  for j in range(N+1):
<<<<<<< HEAD
    for name in ['2X','2Y','4SigmaM']:
=======
    for name in ['2X']:
>>>>>>> 25bfdb4a0a27ebfa182c5df1472ce635d5a718b6
      if(j==0): f='base/'+name+'.vmc.gosling.json'  #GS
      else: f=basename+'/'+name+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc.gosling.json'
      
      data=json.load(open(f,'r'))
      obdm,__=get_qwalk_dm(data['properties']['tbdm_basis'])
      energy=data['properties']['total_energy']['value'][0]*27.2114
      energy_err=data['properties']['total_energy']['error'][0]*27.2114

      n_labels=np.array(["y1","y2","x1","x2","s1","s2","s3"])
<<<<<<< HEAD
      sz_labels=np.array(["sy1","sy2","sx1","sx2","ss1","ss2","ss3"])
      n=np.diag(obdm[0,:,:])+np.diag(obdm[1,:,:])
      sz=np.diag(obdm[0,:,:])-np.diag(obdm[1,:,:])
      
      d=pd.DataFrame(np.array([energy,energy_err]+list(n)+list(sz))[:,np.newaxis].T,columns=['energy','energy_err']+list(n_labels)+list(sz_labels))
=======
      n=np.diag(obdm[0,:,:])+np.diag(obdm[1,:,:])

      d=pd.DataFrame(np.array([energy,energy_err]+list(n))[:,np.newaxis].T,columns=['energy','energy_err']+list(n_labels))
>>>>>>> 25bfdb4a0a27ebfa182c5df1472ce635d5a718b6
      d=d.astype('double')
      d['base_state']=name
      if(df is None): df=d
      else: df=pd.concat((df,d),axis=0)      
  
  fout=basename+'/ex_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_gosling.pickle'
  df.to_pickle(fout)
  return df

if __name__=='__main__':
  detgen='a'
  N=200
  Ndet=10
  gsw=0.9
  basename='run1a'
  df=gather_all(detgen,N,Ndet,gsw,basename)
