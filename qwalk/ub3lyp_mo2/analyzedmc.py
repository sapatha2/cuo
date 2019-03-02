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
  for basestate in range(8):
    for gsw in np.arange(0.1,1.1,0.1):
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
      
      small_df['basestate']=basestate
      if(np.around(gsw,2)==1.0): small_df['basestate']=-1
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0)
  return df

def analyze(df):
  df['gsw']=np.round(df['gsw'],2)
  print(list(df))

  df['n_3dd']=df['t_4_4']+df['t_5_5']
  df['n_3dpi']=df['t_1_1']+df['t_2_2']
  df['n_3dz2']=df['t_3_3']
  df['n_3d']=df['n_3dd']+df['n_3dpi']+df['n_3dz2']
  df['n_2ppi']=df['t_6_6']+df['t_7_7']
  df['n_2pz']=df['t_8_8']
  df['n_2p']=df['n_2ppi']+df['n_2pz']
  df['n_4s']=df['t_9_9']
  df['t_pi']=2*(df['t_1_6']+df['t_2_7'])
  df['t_dz']=2*df['t_3_8']
  df['t_sz']=2*df['t_8_9']
  df['t_ds']=2*df['t_3_9']
  df['n']=df['n_3d']+df['n_4s']+df['n_2p']

  #PAIRPLOTS --------------------------------------------------------------------------
  #sns.pairplot(df,vars=['energy','n'],hue='basestate',markers=['o']+['.']*8)
  #sns.pairplot(df,vars=['energy','n_2ppi','n_2pz','n_3d'],hue='basestate',markers=['o']+['.']*8)
  #sns.pairplot(df,vars=['energy','t_pi','t_dz','t_ds','t_sz'],hue='basestate',markers=['o']+['.']*8)
  #plt.show()
  #exit(0)

  #FITS --------------------------------------------------------------------------
  y=df['energy']
  yerr=df['energy_err']
  #BIGGEST
  #X=df[['n_3dd','n_3d','n_2ppi','n_2pz','n_2p','n_4s','t_pi','t_sz','t_dz','t_ds']]
  #SMALLEST
  #X=df[['n_3d','n_2ppi','n_2pz']]
  #SEQUENTIALLY BETTER
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_sz']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_dz']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_dz','t_sz']]

  #X=df[['n_3d','n_2ppi','n_2pz','t_dz']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_sz']]
  
  X=df[['n_3d','n_2ppi','n_2pz','t_ds','t_pi']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_pi','t_ds']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_dz','t_ds']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_sz','t_ds']]
  #X=df[['n_3d','n_2ppi','n_2pz','t_dz','t_sz','t_ds']]

  X=sm.add_constant(X)
  beta=0.0
  wls=sm.WLS(y,X,weights=np.exp(-beta*(y-min(y)))).fit()
  print(wls.summary())
 
  df['pred']=wls.predict(X)
  df['resid']=df['energy']-df['pred'] 
  sns.pairplot(df,vars=['resid','t_pi','t_dz','t_sz'],hue='basestate',markers=['o']+['.']*8)
  '''
  g = sns.FacetGrid(df,hue='basestate',hue_kws=dict(marker=['o']+['.']*8),palette=sns.color_palette('husl',9))
  g.map(plt.errorbar, "pred", "energy", "energy_err",fmt='o').add_legend()
  plt.plot(df['energy'],df['energy'],'k--')
  '''
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
 
  #dpi,dpi,dz2,dd,dd,ppi,ppi,pz,4s
  #e3d,e2pz,e2ppi,tpi,tdz,tsz=(-3.2487,-2.5759,-1.6910,0.3782,-0.7785,1.1160) #DMC
  e3d,e2pz,e2ppi,tpi,tdz,tsz=(-3.3324,-1.8762,-0.9182,0.8266,0,0) #DMC

  H=np.diag([e3d,e3d,e3d,e3d,e3d,e2ppi,e2ppi,e2pz,0])
  H[0,5]=tpi
  H[5,0]=tpi
  H[1,6]=tpi
  H[6,1]=tpi
  H[2,7]=tdz
  H[7,2]=tdz
  H[7,8]=tsz
  H[8,7]=tsz

  mo_coeff=m.mo_coeff[:,5:14] #Only include active MOs
  a=a[:,[1,5,6,7,8,9,11,12,13]]                 #Only include active IAOs
  e1=reduce(np.dot,(a.T,s,mo_coeff,H,mo_coeff.T,s.T,a))
  e1=(e1+e1.T)/2. 
  
  plt.matshow(e1,vmax=5,vmin=-5,cmap=plt.cm.bwr)
  plt.colorbar()
  #labels=['3s','4s','3px','3py','3pz','3dxy','3dyz','3dz2','3dxz','3dx2-y2','2s','2px','2py','2pz']
  labels=['4s','3dxy','3dyz','3dz2','3dxz','3dx2-y2','2px','2py','2pz']
  plt.xticks(np.arange(len(labels)),labels,rotation=90)
  plt.yticks(np.arange(len(labels)),labels)
  plt.show()

if __name__=='__main__':
  df=collectdf()
  analyze(df)
