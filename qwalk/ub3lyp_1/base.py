import os 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm

######################################################################################
#FROZEN METHODS
#Collect df
def collect_df():
  df=None
  for basestate in range(11):
    for gsw in [1.0]:
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)

      small_df['basestate']=basestate
      small_df['Sz']=0.5
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in range(6):
    for gsw in [1.0]:
      f='../ub3lyp_3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+11
      small_df['Sz']=1.5
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
  df['Up']=df['u6']+df['u7']+df['u8']

  df['Jdd']=np.zeros(df.shape[0])
  orb1=[1,1,1,1,2,2,2,3,3,4]
  orb2=[2,3,4,5,3,4,5,4,5,5]
  for i in range(len(orb1)):
    df['Jdd']+=df['j_'+str(orb1[i])+'_'+str(orb2[i])]
  df['Jsd']=df['j_0_1']+df['j_0_2']+df['j_0_3']+df['j_0_4']+df['j_0_5']
  df['Jsp']=df['j_0_8']+df['j_0_6']+df['j_0_7']
  df['Jpp']=df['j_6_7']+df['j_6_8']+df['j_7_8']
  df['Jdp']=np.zeros(df.shape[0])
  for i in range(1,6):
    df['Jdp']+=df['j_'+str(i)+'_6']+df['j_'+str(i)+'_7']+df['j_'+str(i)+'_8']
  return df

df = collect_df()
df = format_df(df)
df = df.sort_values(by='energy')
df['energy']/=27.2114
df['energy_err']/=27.2114
#terms = ['energy','basestate','Sz','mo_n_3dz2','mo_n_3dpi','mo_n_2ppi','mo_n_2pz','mo_n_4s','mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz','Us']
#terms = ['energy','basestate','Sz','Us','Up','Ud','Jdd','Jpp','Jdp','Jsp','Jsd']
#print(df[terms])

'''
terms = ['mo_n_3dz2','mo_n_3dpi','mo_n_2ppi','mo_n_2pz','mo_n_4s','mo_t_pi','mo_t_dz','mo_t_ds','mo_t_sz',
'Us','Up','Ud','Jdd','Jpp','Jdp','Jsp','Jsd']
var=df[terms].var()
print(var.sort_values())
'''

plt.plot(df['energy'],df['energy_err'],'o')
plt.show()
