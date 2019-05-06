import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm
import pandas as pd
import numpy as np 

#f='run1s/ex_s_Ndet2_gsw0.0_gosling.pickle'
f='run2s/ex_s_Ndet10_gsw0.7_gosling.pickle'
#f='run1a/ex_a_Ndet2_gsw0.0_gosling.pickle'
#f='run2a/ex_a_Ndet10_gsw0.7_gosling.pickle'
df=pd.read_pickle(f)
df['J_cu']=df['J_4s_3d']+df['J_3d']
sns.pairplot(df,vars=['energy','J_4s_3d','J_cu','n_3d','n_4s'],hue='base_state')
plt.show()

