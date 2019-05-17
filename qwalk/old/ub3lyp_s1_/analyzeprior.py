import os 
import numpy as np 
import scipy as sp 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.linear_model
from ed import ED
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce 
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.optimize import linear_sum_assignment 
from find_connect import  *
import matplotlib as mpl 
from prior import prior_fit, prior_score
import itertools 

######################################################################################
#FROZEN METHODS
#Collect df
def collect_df():
  df=None
  for basestate in range(11):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)

      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=0.5
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0,sort=True)
  
  for basestate in range(6):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../ub3lyp_s3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+11
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in range(4):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../../ub3lyp_extra_1/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+17
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
 
  for basestate in range(2):
    for gsw in np.arange(-1.0,1.2,0.2):
      f='../../ub3lyp_extra_3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+21
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)

  df['prior'] = False
 
  #Include all except 2z -> pi/s, since we actually have that variation in the model
  small_df = pd.read_pickle('../../ub3lyp_extra_1/mo_ex/dmc_gosling.pickle').iloc[[0,1,3]]
  small_df['basestate'] = -100
  small_df['prior'] = True
  small_df['Sz'] = 0.5
  df = pd.concat((df,small_df),axis=0,sort=True)

  return df

#Formatting
def format_df(df):
  df['mo_n_3dd']=df['mo_4_4']+df['mo_5_5']
  df['mo_n_3dpi']=df['mo_1_1']+df['mo_2_2']
  df['mo_n_3dz2']=df['mo_3_3']
  df['mo_n_3d']=df['mo_n_3dd']+df['mo_n_3dpi']+df['mo_n_3dz2']
  df['mo_n_2ppi']=df['mo_6_6']+df['mo_7_7']
  df['mo_n_2pz']=df['mo_8_8']
  df['mo_n_2p']=df['mo_n_2ppi']+df['mo_n_2pz']
  df['mo_n_4s']=df['mo_9_9']
  df['mo_t_pi']=2*(df['mo_1_6']+df['mo_2_7'])
  df['mo_t_dz']=2*df['mo_3_8']
  df['mo_t_sz']=2*df['mo_8_9']
  df['mo_t_ds']=2*df['mo_3_9']
  df['mo_n']=df['mo_n_3d']+df['mo_n_2p']+df['mo_n_4s']

  df['Us']=df['u0']
  df['Ud']=df['u1']+df['u2']+df['u3']+df['u4']+df['u5']

  df['Jd']=np.zeros(df.shape[0])
  orb1=[1,1,1,1,2,2,2,3,3,4]
  orb2=[2,3,4,5,3,4,5,4,5,5]
  for i in range(len(orb1)):
    df['Jd']+=df['j_'+str(orb1[i])+'_'+str(orb2[i])]
  df['Jsd']=df['j_0_1']+df['j_0_2']+df['j_0_3']+df['j_0_4']+df['j_0_5']
  df['Jcu']=df['Jsd']+df['Jd']
  return df #format_df_iao(df)

#Get IAO psums
def format_df_iao(df):
  df['beta']=-1000
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]

  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()

  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s']
  #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s

  df['iao_n_3dd']=0
  df['iao_n_3dpi']=0
  df['iao_n_3dz2']=0
  df['iao_n_3d']=0
  df['iao_n_2pz']=0
  df['iao_n_2ppi']=0
  df['iao_n_4s']=0
  df['iao_t_pi']=0
  df['iao_t_dz']=0
  df['iao_t_ds']=0
  df['iao_t_sz']=0
  for z in range(df.shape[0]):
    print(z)
    e=np.zeros((9,9))
    orb1=[1,2,3,4,5,6,7,8,9,1,2,3,8,3]
    orb2=[1,2,3,4,5,6,7,8,9,6,7,8,9,9]
    for i in range(len(orb1)):
      e[orb1[i]-1,orb2[i]-1]=df['mo_'+str(orb1[i])+'_'+str(orb2[i])].values[z]
    mo_to_iao = reduce(np.dot,(mo.T,s,iao))
    e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
    e[np.abs(e)<1e-10]=0
    e=(e+e.T)/2
    df['iao_n_3dd'].iloc[z]=np.sum(np.diag(e)[[0,1]])
    df['iao_n_3dpi'].iloc[z]=np.sum(np.diag(e)[[2,3]])
    df['iao_n_3dz2'].iloc[z]=np.diag(e)[6]
    df['iao_n_3d'].iloc[z]=np.sum(np.diag(e)[[0,1,2,3,6]])
    df['iao_n_2pz'].iloc[z]=np.sum(np.diag(e)[7])
    df['iao_n_2ppi'].iloc[z]=np.sum(np.diag(e)[[4,5]])
    df['iao_n_4s'].iloc[z]=np.sum(np.diag(e)[8])
    df['iao_t_pi'].iloc[z]=2*(e[3,4]+e[2,5])
    df['iao_t_ds'].iloc[z]=2*e[6,8]
    df['iao_t_dz'].iloc[z]=2*e[6,7]
    df['iao_t_sz'].iloc[z]=2*e[7,8]
  return df

######################################################################################
#ANALYSIS CODE 

#ED
def ed(model,exp_parms):
  param_names=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz',
  'mo_t_ds','mo_t_sz','Jsd','Us']
  params=[]
  for parm in param_names:
    if(parm in model): params.append(exp_parms[model.index(parm)])
    else: params.append(0)
  
  norb=9
  nelec=(8,7)
  nroots=30
  res1=ED(params,nroots,norb,nelec)

  nelec=(9,6)
  nroots=30
  res3=ED(params,nroots,norb,nelec)

  E1 = res1[0]
  E3 = res3[0]
  
  return [E1,E3]

#RUN
def analyze(df=None,save=False):
  #Analysis
  '''
  z=-1
  core=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','Us']]
  hopping=df[['mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz']]
  model_list=[list(core)]
  for n in range(1,hopping.shape[1]+1):
    s=set(list(hopping))
    models=list(map(set,itertools.combinations(s,n)))
    model_list+=[list(core)+list(m) for m in models]
  print(len(model_list))

  z=-1
  for model in model_list:
    z+=1
    print(model)
    fit_df = df[model]
    fit_df['y'] = df['energy']
    fit_df['const'] = 1
    fit_df['prior'] = df['prior']

    x = []
    y = []
    for lam in np.arange(0,1.2,0.2):
      parm = prior_fit(fit_df,lam=lam)
      score = prior_score(parm,fit_df)
      print(score)
      x.append(score[0])
      y.append(score[1])
    plt.plot(x,y,'o-',label=str(z))
  plt.ylabel('score_prior')
  plt.xlabel('score_train')
  plt.legend(loc='best')
  plt.show()
  exit(0)
  '''

  model = ['mo_n_4s','mo_n_2ppi','mo_n_2pz','Us']
  fit_df = df[model]
  fit_df['y'] = df['energy']
  fit_df['const'] = 1
  fit_df['prior'] = df['prior']
  lam = 0.2
  parm = prior_fit(fit_df,lam=lam)
  print(parm)
  score = prior_score(parm,fit_df)
  print(score)

  fit_df = fit_df[fit_df['prior']==False]
  yhat = np.dot(parm,fit_df.drop(columns=['y','prior']).values.T)
  plt.plot(yhat,fit_df['y'],'o')
  plt.plot(fit_df['y'],fit_df['y'],'--k')
  plt.show()

  e1,e3=ed(model,parm[:-1])
  ref = min(e1)
  e1 = np.array(e1) - ref
  e3 = np.array(e3) - ref
  plt.plot(e1,'o')
  plt.plot(e3,'o')
  plt.show()
if __name__=='__main__':
  #DATA COLLECTION
  '''
  df=collect_df()
  df=format_df(df)
  df=format_df_iao(df)
  df.to_pickle('prior_gosling.pickle')
  exit(0)
  '''

  df = pd.read_pickle('prior_gosling.pickle')
  analyze(df)
