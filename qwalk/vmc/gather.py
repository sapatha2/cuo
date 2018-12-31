import json 
import pandas as pd
import numpy as np 
import sys
sys.path.append('../../../downfolding/')
from shivesh_downfold_tools import get_qwalk_dm, sum_onebody, sum_J, sum_U, sum_V

full_labels=np.array(["4s","3dxy","3dyz","3dz2","3dxz","3dx2y2","2px","2py","2pz"])

def gather_all(detgen,N,Ndet,gsw,basename):
  ''' 
  Gathers all your data and stores into 
  '''
  df=None
  for j in range(N+1):
    for state in ['','2']:
      if(j==0): f='base/gs'+state+'.vmc_tbdm.gosling.json'  #GS
      else: f=basename+'/ex'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc.gosling.json'
      
      data=json.load(open(f,'r'))
      obdm,__,tbdm,__=get_qwalk_dm(data['properties']['tbdm_basis'])
      energy=data['properties']['total_energy']['value'][0]*27.2114
      energy_err=data['properties']['total_energy']['error'][0]*27.2114
 
      one_labels=['n_'+x for x in full_labels]+['4s-3dz2','4s-2pz','3dz2-2pz','3dyz-2py','3dxz-2px']
      orb1=[0,1,2,3,4,5,6,7,8,0,0,3,2,4]
      orb2=[0,1,2,3,4,5,6,7,8,3,8,8,7,6]
      one_body=sum_onebody(obdm,orb1,orb2)
      
      U_labels=['U_'+x for x in full_labels]
      orb=np.arange(9)
      U=sum_U(tbdm,orb)

      orb1=[1,1,1,1,2,2,2,3,3,4]
      orb2=[2,3,4,5,3,4,5,4,5,5]
      V_labels=['V_'+full_labels[orb1[i]]+'_'+full_labels[orb2[i]] for i in range(len(orb1))]
      V=sum_V(tbdm,orb1,orb2) 

      orb1=[1,1,1,1,2,2,2,3,3,4]
      orb2=[2,3,4,5,3,4,5,4,5,5]
      J_labels=['J_'+full_labels[orb1[i]]+'_'+full_labels[orb2[i]] for i in range(len(orb1))]
      J=sum_J(tbdm,orb1,orb2) 

      d=pd.DataFrame(np.array([energy,energy_err]+list(one_body)+list(U)+list(V)+list(J))[:,np.newaxis].T,columns=['energy','energy_err']+one_labels+U_labels+V_labels+J_labels)
      if(df is None): df=d
      else: df=pd.concat((df,d),axis=0)      
  
  df['n_3dd']=df['n_3dxy']+df['n_3dx2y2']
  df['n_3dpi']=df['n_3dxz']+df['n_3dyz']
  df['n_2ppi']=df['n_2px']+df['n_2py']
  df['n_3d']=df['n_3dz2']+df['n_3dd']+df['n_3dpi']
  df['n_2p']=df['n_2pz']+df['n_2ppi']
  df['3dpi-2ppi']=df['3dxz-2px']+df['3dyz-2py']
  df=df.drop(columns=['n_3dxy','n_3dx2y2','n_3dxz','n_3dyz','n_2py','n_2px','3dxz-2px','3dyz-2py'])

  df['U_3dd']=df['U_3dxy']+df['U_3dx2y2']
  df['U_3dpi']=df['U_3dxz']+df['U_3dyz']
  df['U_2ppi']=df['U_2px']+df['U_2py']
  df['U_3d']=df['U_3dz2']+df['U_3dd']+df['U_3dpi']
  df['U_2p']=df['U_2pz']+df['U_2ppi']
  df=df.drop(columns=['U_3dxy','U_3dx2y2','U_3dxz','U_3dyz','U_2py','U_2px'])
  
  df['V_3dd']=df['V_3dxy_3dx2y2']
  df['V_3dd_3dz2']=df['V_3dxy_3dz2']+df['V_3dz2_3dx2y2']
  df['V_3dpi']=df['V_3dyz_3dxz']
  df['V_3dpi_3dz2']=df['V_3dyz_3dz2']+df['V_3dz2_3dxz']
  df['V_3dpi_3dd']=df['V_3dxy_3dxz']+df['V_3dxy_3dyz']+df['V_3dxz_3dx2y2']+df['V_3dyz_3dx2y2']
  df['V_3d']=df['V_3dd']+df['V_3dd_3dz2']+df['V_3dpi']+df['V_3dpi_3dz2']+df['V_3dpi_3dd']
  df=df.drop(columns=['V_3dxy_3dx2y2','V_3dxy_3dz2','V_3dz2_3dx2y2','V_3dyz_3dxz',
  'V_3dyz_3dz2','V_3dz2_3dxz','V_3dxy_3dxz','V_3dxy_3dyz','V_3dxz_3dx2y2','V_3dyz_3dx2y2'])

  df['J_3dd']=df['J_3dxy_3dx2y2']
  df['J_3dd_3dz2']=df['J_3dxy_3dz2']+df['J_3dz2_3dx2y2']
  df['J_3dpi']=df['J_3dyz_3dxz']
  df['J_3dpi_3dz2']=df['J_3dyz_3dz2']+df['J_3dz2_3dxz']
  df['J_3dpi_3dd']=df['J_3dxy_3dxz']+df['J_3dxy_3dyz']+df['J_3dxz_3dx2y2']+df['J_3dyz_3dx2y2']
  df['J_3d']=df['J_3dd']+df['J_3dd_3dz2']+df['J_3dpi']+df['J_3dpi_3dz2']+df['J_3dpi_3dd']
  df=df.drop(columns=['J_3dxy_3dx2y2','J_3dxy_3dz2','J_3dz2_3dx2y2','J_3dyz_3dxz',
  'J_3dyz_3dz2','J_3dz2_3dxz','J_3dxy_3dxz','J_3dxy_3dyz','J_3dxz_3dx2y2','J_3dyz_3dx2y2'])

  fout=basename+'/ex'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
if __name__=='__main__':
  detgen='a'
  N=50
  Ndet=10
  gsw=0.8
  basename='run2a'
  df=gather_all(detgen,N,Ndet,gsw,basename)
