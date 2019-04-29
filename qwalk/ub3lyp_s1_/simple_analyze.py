import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.feature_selection import RFE, RFECV
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression, base
from sklearn.model_selection import cross_val_score
from ed import h1_moToIAO
import seaborn as sns 
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from ed import ED

def calc_density(df,dE):
  density=np.zeros(df.shape[0])
  dE = 0.1 #eV
  for p in range(df.shape[0]):
    density[p] = np.sum((df['energy']<(df['energy'].iloc[p] + dE))&(df['energy']>(df['energy'].iloc[p] - dE)))
  df['density']=density

  return df

def diagonalize(params):
  norb=9
  nelec=(8,7)
  nroots=25
  res1=ED(params,nroots,norb,nelec)

  nelec=(9,6)
  nroots=15
  res3=ED(params,nroots,norb,nelec)

  E = res1[0]
  Sz = np.ones(len(E))*0.5
  dm = res1[2] + res1[3]
  n_3d = dm[:,0,0]+dm[:,1,1]+dm[:,2,2]+dm[:,3,3]+dm[:,6,6]
  n_2ppi = dm[:,4,4]+dm[:,5,5]
  n_2pz = dm[:,7,7]
  n_4s = dm[:,8,8]
  t_pi = 2*(dm[:,3,4]+dm[:,2,5])
  t_ds = 2*dm[:,6,8]
  t_dz = 2*dm[:,6,7]
  t_sz = 2*dm[:,7,8]
  d = pd.DataFrame({'energy':E,'Sz':Sz,'iao_n_3d':n_3d,'iao_n_2pz':n_2pz,'iao_n_2ppi':n_2ppi,'iao_n_4s':n_4s,
  'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz,'iao_Us':res1[4],'iao_Jsd':res1[5]})

  E = res3[0]
  Sz = np.ones(len(E))*1.5
  dm = res3[2] + res3[3]
  n_3d = dm[:,0,0]+dm[:,1,1]+dm[:,2,2]+dm[:,3,3]+dm[:,6,6]
  n_2ppi = dm[:,4,4]+dm[:,5,5]
  n_2pz = dm[:,7,7]
  n_4s = dm[:,8,8]
  t_pi = 2*(dm[:,3,4]+dm[:,2,5])
  t_ds = 2*dm[:,6,8]
  t_dz = 2*dm[:,6,7]
  t_sz = 2*dm[:,7,8]
  d = pd.concat((d,pd.DataFrame({'energy':E,'Sz':Sz,'iao_n_3d':n_3d,'iao_n_2pz':n_2pz,'iao_n_2ppi':n_2ppi,'iao_n_4s':n_4s,
  'iao_t_pi':t_pi,'iao_t_ds':t_ds,'iao_t_dz':t_dz,'iao_t_sz':t_sz,'iao_Us':res3[4],'iao_Jsd':res3[5]})),axis=0)

  d['energy']-=min(d['energy'])
  return d

######################################################################################
#RUN
def analyze(df,save=False):
 

  pca_parms=['mo_t_pi','mo_t_ds','mo_t_dz','mo_t_sz']
  pca=PCA(n_components=len(pca_parms))
  pca.fit(df[pca_parms])
  print(pca.explained_variance_ratio_)
  Xbar=df[pca_parms]
  #Xbar=pca.transform(Xbar)
  Xbar = np.dot(Xbar,pca.components_.T)

  for i in range(len(pca_parms)):
    df['x'+str(i)]=Xbar[:,i]

  parms=['mo_n_3d','mo_n_2ppi','mo_n_2pz','Us','x0','x1']#,'x2','x3']
  X=df[parms]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.OLS(y,X).fit()
  print(ols.summary())


  print(ols.params)
  pca_params = ols.params[-2:].values
  pca_params = list(pca_params) + [0,0]
 
  reg_params = np.dot(pca_params,pca.components_)
  print(pca_params)
  print(reg_params)

  exit(0)

  df['resid']=df['energy']-ols.predict()
  #model=['mo_n_3d','mo_n_4s','mo_n_2ppi','mo_n_2pz','Us']
  #model=['mo_t_pi','mo_t_ds','mo_t_dz','mo_t_sz']
  sns.pairplot(df,vars=['resid','x0','x1','x2','x3'])#,vars=['resid']+model)
  plt.show()
  exit(0)

  beta=2

  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Us']
  X=df[model]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.WLS(y,X,np.exp(-beta*(y-min(y)))).fit()
  params = ols.params[1:]
  ed = diagonalize(list(params[:-2])+[params[-2],0,0,0,0,params[-1]])
  print(ols.summary())
  print(ed)
  
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_dz','Us']
  X=df[model]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.WLS(y,X,np.exp(-beta*(y-min(y)))).fit()
  params = ols.params[1:]
  ed = diagonalize(list(params[:-2])+[0,params[-2],0,0,0,params[-1]])
  print(ols.summary())
  print(ed)
  
  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_sz','Us']
  X=df[model]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.WLS(y,X,np.exp(-beta*(y-min(y)))).fit()
  params = ols.params[1:]
  ed = diagonalize(list(params[:-2])+[0,0,params[-2],0,0,params[-1]])
  print(ols.summary())
  print(ed)

  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_ds','Us']
  X=df[model]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.WLS(y,X,np.exp(-beta*(y-min(y)))).fit()
  params = ols.params[1:]
  ed = diagonalize(list(params[:-2])+[0,0,0,params[-2],0,params[-1]])
  print(ols.summary())
  print(ed)

  model=['mo_n_4s','mo_n_2ppi','mo_n_2pz','Us']
  X=df[model]
  X=sm.add_constant(X)
  y=df['energy']
  ols=sm.WLS(y,X,np.exp(-beta*(y-min(y)))).fit()
  params = ols.params[1:]
  params = list(params[:3])+[0]+list(params[3:-1])+[0,0,0,0,params[-1]]
  ed = diagonalize(params)
  print(ols.summary())  
  print(ed)

  exit(0)

if __name__=='__main__':
  #DATA COLLECTION
  #df=collect_df()
  #df=format_df(df)
  #df.to_pickle('formatted_gosling.pickle')
  #exit(0)

  #DATA ANALYSIS
  df=pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
