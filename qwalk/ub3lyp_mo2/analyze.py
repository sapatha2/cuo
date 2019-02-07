import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV 
from sklearn.model_selection import cross_val_score
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
from functools import reduce 

def collectdf():
  df=None
  for basestate in range(3):
    for gsw in np.arange(0.1,1.01,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/gosling.pickle' 
      small_df=pd.read_pickle(f)
      
      small_df['basestate']=basestate
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  df['gsw']=np.round(df['gsw'],2)
  
  df['n_3dd']=df['t_8_8']+df['t_9_9']
  df['n_3dpi']=df['t_5_5']+df['t_6_6']
  df['n_3dz2']=df['t_7_7']
  df['n_3d']=df['n_3dd']+df['n_3dpi']+df['n_3dz2']
  df['n_2ppi']=df['t_10_10']+df['t_11_11']
  df['n_2pz']=df['t_12_12']
  df['n_2p']=df['n_2ppi']+df['n_2pz']
  df['n_4s']=df['t_13_13']
  df['t_pi']=2*(df['t_5_10']+df['t_6_11'])
  df['t_dz']=2*df['t_7_12']
  df['t_ds']=2*df['t_7_13']
  df['t_sz']=2*df['t_12_13']

  #PAIRPLOTS --------------------------------------------------------------------------
  #sns.pairplot(df,vars=['energy','n_3dd','n_3dpi','n_3dz2','n_3d'],hue='basestate',markers=['o']+['.']*8)
  #sns.pairplot(df,vars=['energy','n_2ppi','n_2pz','n_2p','n_4s'],hue='basestate',markers=['o']+['.']*8)
  #sns.pairplot(df,vars=['energy','t_pi','t_dz','t_ds','t_sz'],hue='basestate',markers=['o']+['.']*8)
  #plt.show()
  #exit(0)

  #FITS --------------------------------------------------------------------------
  y=df['energy']
  yerr=df['energy_err']
  #BIGGEST
  #X=df[['n_3dd','n_3d','n_2ppi','n_2pz','n_2p','n_4s','t_pi','t_sz','t_dz','t_ds']]
  #SMALLEST
  X=df[['n_3d','n_2ppi','n_2pz']]
  #SEQUENTIALLY BETTER
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_ds']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_dz','t_sz']]

  X=sm.add_constant(X)
  ols=sm.OLS(y,X).fit()
  print(ols.params)
  print(ols.summary())
  
  df['pred']=ols.predict(X)
  g = sns.FacetGrid(df,hue='basestate',hue_kws=dict(marker=['o']+['.']*8))
  g.map(plt.errorbar, "pred", "energy", "energy_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  plt.show()
  exit(0)

  #ROTATE TO IAOS --------------------------------------------------------------------------
  #Gather IAOs
  f='../../pyscf/analyze/b3lyp_iao_b.pickle' #IAOs which span the MOs
  a=np.load(f)

  chkfile='../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk' #MOs we used to calculate RDM elements
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  s=m.get_ovlp()
  H1=np.diag([-3.1957,-3.1957,-3.1957,-3.1957,-3.1957,-1.6972,-1.6972,-2.5474,0])
  H1[0,5]=0.7858
  H1[5,0]=0.7858
  H1[1,6]=0.7858
  H1[6,1]=0.7858
  H1[2,7]=-0.8329
  H1[7,2]=-0.8329
  H1[7,8]=1.6742
  H1[8,7]=1.6742

  mo_coeff=m.mo_coeff[:,5:14]
  e1=reduce(np.dot,(a.T,s,mo_coeff,H1,mo_coeff.T,s.T,a))
  e1=(e1+e1.T)/2. 
  
  plt.matshow(e1,vmax=5,vmin=-5,cmap=plt.cm.bwr)
  plt.colorbar()
  labels=['3s','4s','3px','3py','3pz','3dxy','3dyz','3dz2','3dxz','3dx2-y2','2s','2px','2py','2pz']
  plt.xticks(np.arange(14),labels,rotation=90)
  plt.yticks(np.arange(14),labels)
  plt.show()

if __name__=='__main__':
  df=collectdf()
  analyze(df)
