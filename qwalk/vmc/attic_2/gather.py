import json 
import pandas as pd
import numpy as np 
import sys
sys.path.append('../../../downfolding/')
from shivesh_downfold_tools import get_qwalk_dm, sum_onebody, sum_J, sum_U, sum_V

full_labels=np.array(["4s","3dxy","3dyz","3dz2","3dxz","3dx2y2","2px","2py","2pz"])
name={'':'2X','2':'4SigmaM','3':'2Y'}
def gather_all(detgen,N,Ndet,gsw,basename):
  ''' 
  Gathers all your data and stores into 
  '''
  df=None
  for j in range(N+1):
    for state in ['','2','3']:
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

      orb1=[0,0,0,0,0,1,1,1,1,2,2,2,3,3,4,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
      orb2=[1,2,3,4,5,2,3,4,5,3,4,5,4,5,5,6,7,8,6,7,8,6,7,8,6,7,8,6,7,8,6,7,8]
      V_labels=['V_'+full_labels[orb1[i]]+'_'+full_labels[orb2[i]] for i in range(len(orb1))]
      V=sum_V(tbdm,orb1,orb2) 

      orb1=[0,0,0,0,0,1,1,1,1,2,2,2,3,3,4,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
      orb2=[1,2,3,4,5,2,3,4,5,3,4,5,4,5,5,6,7,8,6,7,8,6,7,8,6,7,8,6,7,8,6,7,8]
      J_labels=['J_'+full_labels[orb1[i]]+'_'+full_labels[orb2[i]] for i in range(len(orb1))]
      J=sum_J(tbdm,orb1,orb2) 

      d=pd.DataFrame(np.array([energy,energy_err]+list(one_body)+list(U)+list(V)+list(J))[:,np.newaxis].T,columns=['energy','energy_err']+one_labels+U_labels+V_labels+J_labels)
      d=d.astype('double')
      d['base_state']=name[state]
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
  df['V_4s_3dd']=df['V_4s_3dxy']+df['V_4s_3dx2y2']
  df['V_4s_3dpi']=df['V_4s_3dxz']+df['V_4s_3dyz']
  df['V_4s_3d']=df['V_4s_3dd']+df['V_4s_3dpi']+df['V_4s_3dz2']
  df['V_4s_2ppi']=df['V_4s_2px']+df['V_4s_2py']
  df['V_4s_2p']=df['V_4s_2ppi']+df['V_4s_2pz']
  df['V_3dpi_2pz']=df['V_3dxz_2pz']+df['V_3dyz_2pz']
  df['V_3dd_2pz']=df['V_3dxy_2pz']+df['V_3dx2y2_2pz']
  df['V_3dpi_2ppi']=df['V_3dxz_2px']+df['V_3dxz_2py']+df['V_3dyz_2px']+df['V_3dyz_2py']
  df['V_3dd_2ppi']=df['V_3dxy_2px']+df['V_3dxy_2py']+df['V_3dx2y2_2px']+df['V_3dx2y2_2py']
  df['V_3dz2_2ppi']=df['V_3dz2_2px']+df['V_3dz2_2py']
  df['V_3d_2p']=df['V_3dpi_2pz']+df['V_3dd_2pz']+df['V_3dz2_2pz']+df['V_3dpi_2ppi']+df['V_3dd_2ppi']+df['V_3dz2_2ppi']
  df=df.drop(columns=['V_3dxy_3dx2y2','V_3dxy_3dz2','V_3dz2_3dx2y2','V_3dyz_3dxz',
  'V_3dyz_3dz2','V_3dz2_3dxz','V_3dxy_3dxz','V_3dxy_3dyz','V_3dxz_3dx2y2','V_3dyz_3dx2y2',
  'V_4s_3dxy','V_4s_3dx2y2','V_4s_3dxz','V_4s_3dyz','V_4s_2px','V_4s_2py',
  'V_3dxz_2pz','V_3dyz_2pz','V_3dxy_2pz','V_3dx2y2_2pz','V_3dxz_2px','V_3dxz_2py','V_3dyz_2px','V_3dyz_2pz',
  'V_3dxy_2px','V_3dxy_2py','V_3dx2y2_2px','V_3dx2y2_2py','V_3dz2_2px','V_3dz2_2py'])

  df['J_3dd']=df['J_3dxy_3dx2y2']
  df['J_3dd_3dz2']=df['J_3dxy_3dz2']+df['J_3dz2_3dx2y2']
  df['J_3dpi']=df['J_3dyz_3dxz']
  df['J_3dpi_3dz2']=df['J_3dyz_3dz2']+df['J_3dz2_3dxz']
  df['J_3dpi_3dd']=df['J_3dxy_3dxz']+df['J_3dxy_3dyz']+df['J_3dxz_3dx2y2']+df['J_3dyz_3dx2y2']
  df['J_3d']=df['J_3dd']+df['J_3dd_3dz2']+df['J_3dpi']+df['J_3dpi_3dz2']+df['J_3dpi_3dd']
  df['J_4s_3dd']=df['J_4s_3dxy']+df['J_4s_3dx2y2']
  df['J_4s_3dpi']=df['J_4s_3dxz']+df['J_4s_3dyz']
  df['J_4s_3d']=df['J_4s_3dd']+df['J_4s_3dpi']+df['J_4s_3dz2']
  df['J_4s_2ppi']=df['J_4s_2px']+df['J_4s_2py']
  df['J_4s_2p']=df['J_4s_2ppi']+df['J_4s_2pz']
  df['J_3dpi_2pz']=df['J_3dxz_2pz']+df['J_3dyz_2pz']
  df['J_3dd_2pz']=df['J_3dxy_2pz']+df['J_3dx2y2_2pz']
  df['J_3dpi_2ppi']=df['J_3dxz_2px']+df['J_3dxz_2py']+df['J_3dyz_2px']+df['J_3dyz_2py']
  df['J_3dd_2ppi']=df['J_3dxy_2px']+df['J_3dxy_2py']+df['J_3dx2y2_2px']+df['J_3dx2y2_2py']
  df['J_3dz2_2ppi']=df['J_3dz2_2px']+df['J_3dz2_2py']
  df['J_3d_2p']=df['J_3dpi_2pz']+df['J_3dd_2pz']+df['J_3dz2_2pz']+df['J_3dpi_2ppi']+df['J_3dd_2ppi']+df['J_3dz2_2ppi']
  df=df.drop(columns=['J_3dxy_3dx2y2','J_3dxy_3dz2','J_3dz2_3dx2y2','J_3dyz_3dxz',
  'J_3dyz_3dz2','J_3dz2_3dxz','J_3dxy_3dxz','J_3dxy_3dyz','J_3dxz_3dx2y2','J_3dyz_3dx2y2',
  'J_4s_3dxy','J_4s_3dx2y2','J_4s_3dxz','J_4s_3dyz','J_4s_2px','J_4s_2py',
  'J_3dxz_2pz','J_3dyz_2pz','J_3dxy_2pz','J_3dx2y2_2pz','J_3dxz_2px','J_3dxz_2py','J_3dyz_2px','J_3dyz_2pz',
  'J_3dxy_2px','J_3dxy_2py','J_3dx2y2_2px','J_3dx2y2_2py','J_3dz2_2px','J_3dz2_2py'])

  fout=basename+'/ex_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_gosling.pickle'
  df.to_pickle(fout)
  return df

import matplotlib.pyplot as plt
import statsmodels.api as sm
if __name__=='__main__':
  detgen='s'
  N=50
  Ndet=10
  gsw=0.7
  basename='run1s'
  df=gather_all(detgen,N,Ndet,gsw,basename)
