import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pandas as pd
import numpy as np 

#flist=['run1s/s_Ndet10_gsw0.7_df.pickleR','run2s/s_Ndet10_gsw0.7_df.pickleR']
#flist=['run2a/a_Ndet10_gsw0.8_df.pickleR','run3a/a_Ndet10_gsw0.8_df.pickleR']
flist=['run2a/a_Ndet10_gsw0.8_df.pickleR']
df=None
for fname in flist:
  small=pd.read_pickle(fname)
  small['detgen']=[fname.split("/")[1].split("_")[0]]*small.shape[0]
  small['gsw']=[fname.split("_")[2]]*small.shape[0]
  if(df is None): df=small
  else: df=pd.concat((df,small))

y=df['energy']
X=df.drop(columns=['energy','energy_err','gsw','detgen','obdm_4s_3dz2','obdm_3dz2_3dz2','obdm_3dpi_3dpi'])
for zz in list(X):
  if('tbdm' in zz): X=df.drop(columns=zz)
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())

plt.ylabel('E_VMC (eV)')
plt.xlabel('E_Pred (eV)')
plt.errorbar(ols.predict(X),y,yerr=df['energy_err'],fmt='bo')
plt.plot(y,y,'g--')
plt.title('Singles space 1-body fit')
#plt.savefig('s_Ndet10_gsw0.7_df.pdf')
#plt.show()
