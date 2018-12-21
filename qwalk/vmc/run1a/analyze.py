import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pandas as pd

fname='a_Ndet10_gsw0.9_df.pickleR'
df=pd.read_pickle(fname)

#OLS
y=df['energy']
X=df.drop(columns=['energy','energy_err','obdm_3dpi_3dpi'])
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())

plt.ylabel('E_VMC (eV)')
plt.xlabel('E_Pred (eV)')
plt.errorbar(ols.predict(X),y,yerr=df['energy_err'],fmt='bo')
plt.plot(y,y,'g--')
plt.title('Full active space 1-body fit')
plt.savefig('a_Ndet10_gsw0.9_df.pdf')
