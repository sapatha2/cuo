import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df=pd.read_pickle('s_Ndet10_gsw0.7_df.pickle')
print(df.shape)

#Energy check
e=df['energy']*27.2114 #eV
e-=np.min(e)
plt.hist(e,bins=20)
plt.show()


