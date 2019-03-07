import pandas as pd
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_pickle('c.pickle')
'''
df['n_d']=df['t_5_5']+df['t_6_6']+df['t_7_7']+df['t_8_8']+df['t_9_9']
df['n_pi']=df['t_11_11']+df['t_12_12']
df['n_pz']=df['t_13_13']
df['t_pi']=df['t_6_12']+df['t_8_11']
sns.pairplot(df,vars=['sigU','sigV','n_d','n_pi','n_pz','t_pi','t_1_7','t_1_13','t_7_13'])
plt.show()'''
plt.plot(df.iloc[10:].var(),'ro')
plt.plot(df.iloc[:10].var(),'b.')
plt.show()
