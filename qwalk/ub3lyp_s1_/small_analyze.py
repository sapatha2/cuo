import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.feature_selection import RFE, RFECV
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression, base
from sklearn.model_selection import cross_val_score
from ed import h1_moToIAO
import seaborn as sns 

######################################################################################
#RUN
def analyze(df,save=False):
  lim=(df[df['basestate']==-1])
  lim['energy']-=min(lim['energy'])
  lim=lim.sort_values(by='energy')
  print(lim[['energy','Sz','mo_n_2pz','mo_n_2ppi','mo_n_4s','mo_t_pi','mo_t_sz','mo_t_ds','mo_t_dz']])
  sns.pairplot(lim,vars=['energy','mo_n_2pz','mo_n_2ppi','mo_n_4s','mo_t_pi'],hue='Sz')
  plt.show()
  exit(0)

  #BIGGEST MODEL
  y=df['energy']
  #X=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','Us']]
  X=df[['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','Us']]
  X=sm.add_constant(X)

  ols = sm.OLS(y,X).fit()
  print(ols.summary())

  parms=list(ols.params[1:-1])+[0,0,0]
  parms[3]*=-1
  h1_moToIAO(parms,printvals=True)


if __name__=='__main__':
  #DATA COLLECTION
  #df=collect_df()
  #df=format_df(df)
  #df.to_pickle('formatted_gosling.pickle')
  #exit(0)

  #DATA ANALYSIS
  df=pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
