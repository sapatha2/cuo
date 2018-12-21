import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pandas as pd

fname='s_Ndet10_gsw0.7_df.pickleR'
df=pd.read_pickle(fname)

#OLS
y=df['energy']
X=df[['obdm_4s_4s','obdm_4s_2pz','obdm_2ppi_2ppi','obdm_3dpi_2ppi','obdm_3dz2_2pz']]
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())

plt.ylabel('E_VMC (eV)')
plt.xlabel('E_Pred (eV)')
plt.errorbar(ols.predict(X),y,yerr=df['energy_err'],fmt='bo')
plt.plot(y,y,'g--')
plt.title('Full active space 1-body fit')
plt.savefig('s_Ndet10_gsw0.7_df.pdf')
