import pandas as pd 
import numpy as np 

flist=['run1s/s_Ndet10_gsw0.7_gosling.pickle','run2a/a_Ndet10_gsw0.8_gosling.pickle']
df=None
for f in flist:
  if(df is None): df=pd.read_pickle(f)
  else: df=pd.concat((df,pd.read_pickle(f)),axis=0)
labels=["4s","3dxy","3dyz","3dz2","3dxz","3dx2y2","2px","2py","2pz"]

#Energy units
df['energy']*=27.2114 #eV
df['energy_err']*=27.2114

#1-body 
nu=['obdm_up_'+i+'_'+j for i in labels for j in labels]
nd=['obdm_down_'+i+'_'+j for i in labels for j in labels]
ntot=['obdm_'+i+'_'+j for i in labels for j in labels]
n=df[nu].values+df[nd].values
df[ntot]=pd.DataFrame(n,index=df.index)
df=df.drop(columns=nd+nu)
n=['obdm_'+str(i)+'_'+str(i) for i in labels]
t=['obdm_4s_2pz','obdm_4s_3dz2','obdm_3dyz_2py','obdm_3dz2_2pz','obdm_3dxz_2px']

#Us
Ulabels=['tbdm_updown_%s_%s_%s_%s'%(s,s,s,s) for s in labels]

#Vs
Vlabels=[]
for i in range(len(labels)):
  for j in range(i+1,len(labels)):
    curr_label='tbdm_V_%s_%s'%(labels[i],labels[j])
    df[curr_label]=0
    Vlabels.append(curr_label)
    for s in ['upup','updown','downup','downdown']:
      l='tbdm_'+s+'_%s_%s_%s_%s'%(labels[i],labels[j],labels[i],labels[j])
      df[curr_label]-=df[l]

#Reduction
rdf=df[['energy']+['energy_err']+n+t+Ulabels+Vlabels]

#1-body
rdf['obdm_3dpi_3dpi']=rdf['obdm_3dxz_3dxz']+rdf['obdm_3dyz_3dyz']
rdf['obdm_3dd_3dd']=rdf['obdm_3dxy_3dxy']+rdf['obdm_3dx2y2_3dx2y2']
rdf['obdm_2ppi_2ppi']=rdf['obdm_2px_2px']+rdf['obdm_2py_2py']
rdf['obdm_3dpi_2ppi']=rdf['obdm_3dxz_2px']+rdf['obdm_3dyz_2py']
rdf['obdm_3dz2_2pz']*=-1  #Get the sign structure correct
rdf['obdm_4s_2pz']*=-1    #Get the sign structure correct
rdf[['obdm_4s_2pz','obdm_4s_3dz2','obdm_3dz2_2pz','obdm_3dpi_2ppi']]*=2 #Hermitian conjugate
rdf=rdf.drop(columns=['obdm_3dxz_3dxz','obdm_3dyz_3dyz',
                      'obdm_3dxy_3dxy','obdm_3dx2y2_3dx2y2','obdm_2px_2px',
                      'obdm_2py_2py','obdm_3dxz_2px','obdm_3dyz_2py'])
#2-body
#U
rdf['tbdm_updown_3dd_3dd_3dd_3dd']=rdf['tbdm_updown_3dxy_3dxy_3dxy_3dxy']+rdf['tbdm_updown_3dx2y2_3dx2y2_3dx2y2_3dx2y2']+\
                                   df['tbdm_updown_3dxy_3dx2y2_3dxy_3dx2y2']+df['tbdm_updown_3dx2y2_3dxy_3dx2y2_3dxy']
rdf['tbdm_updown_3dpi_3dpi_3dpi_3dpi']=rdf['tbdm_updown_3dxz_3dxz_3dxz_3dxz']+rdf['tbdm_updown_3dyz_3dyz_3dyz_3dyz']+\
                                       df['tbdm_updown_3dxz_3dyz_3dxz_3dyz']+df['tbdm_updown_3dyz_3dxz_3dyz_3dxz']
rdf['tbdm_updown_2ppi_2ppi_2ppi_2ppi']=rdf['tbdm_updown_2px_2px_2px_2px']+rdf['tbdm_updown_2py_2py_2py_2py']+\
                                       df['tbdm_updown_2px_2py_2px_2py']+df['tbdm_updown_2py_2px_2py_2px']
rdf['tbdm_updown_3d_3d_3d_3d']=-1*(rdf['tbdm_updown_3dd_3dd_3dd_3dd']+rdf['tbdm_updown_3dpi_3dpi_3dpi_3dpi']+rdf['tbdm_updown_3dz2_3dz2_3dz2_3dz2'])
rdf['tbdm_updown_2p_2p_2p_2p']=-1*(rdf['tbdm_updown_2ppi_2ppi_2ppi_2ppi']+rdf['tbdm_updown_2pz_2pz_2pz_2pz'])
rdf=rdf.drop(columns=['tbdm_updown_3dxy_3dxy_3dxy_3dxy','tbdm_updown_3dx2y2_3dx2y2_3dx2y2_3dx2y2',
                      'tbdm_updown_3dxz_3dxz_3dxz_3dxz','tbdm_updown_3dyz_3dyz_3dyz_3dyz',
                      'tbdm_updown_2px_2px_2px_2px','tbdm_updown_2py_2py_2py_2py'])

#V
rdf['tbdm_V_4s_2ppi']=rdf['tbdm_V_4s_2px']+rdf['tbdm_V_4s_2py']
rdf['tbdm_V_4s_2p']=rdf['tbdm_V_4s_2ppi']+rdf['tbdm_V_4s_2pz']
rdf['tbdm_V_4s_3dpi']=rdf['tbdm_V_4s_3dxz']+rdf['tbdm_V_4s_3dyz']
rdf['tbdm_V_4s_3dd']=rdf['tbdm_V_4s_3dx2y2']+rdf['tbdm_V_4s_3dxy']
rdf['tbdm_V_4s_3d']=rdf['tbdm_V_4s_3dpi']+rdf['tbdm_V_4s_3dd']+rdf['tbdm_V_4s_3dz2']

rdf['tbdm_V_3dd_2ppi']=rdf['tbdm_V_3dxy_2px']+rdf['tbdm_V_3dxy_2py']+rdf['tbdm_V_3dx2y2_2px']+rdf['tbdm_V_3dx2y2_2py']
rdf['tbdm_V_3dd_2pz']=rdf['tbdm_V_3dxy_2pz']+rdf['tbdm_V_3dx2y2_2pz']
rdf['tbdm_V_3dd_2p']=rdf['tbdm_V_3dd_2ppi']+rdf['tbdm_V_3dd_2pz']
rdf['tbdm_V_3dd_3dpi']=rdf['tbdm_V_3dxy_3dxz']+rdf['tbdm_V_3dxy_3dyz']+rdf['tbdm_V_3dxz_3dx2y2']+rdf['tbdm_V_3dyz_3dx2y2']
rdf['tbdm_V_3dd_3dz2']=rdf['tbdm_V_3dxy_3dz2']+rdf['tbdm_V_3dz2_3dx2y2']

rdf['tbdm_V_3dpi_2ppi']=rdf['tbdm_V_3dyz_2px']+rdf['tbdm_V_3dyz_2py']+rdf['tbdm_V_3dxz_2px']+rdf['tbdm_V_3dxz_2py']
rdf['tbdm_V_3dpi_2pz']=rdf['tbdm_V_3dyz_2pz']+rdf['tbdm_V_3dxz_2pz']
rdf['tbdm_V_3dpi_2p']=rdf['tbdm_V_3dpi_2ppi']+rdf['tbdm_V_3dpi_2pz']
rdf['tbdm_V_3dpi_3dz2']=rdf['tbdm_V_3dyz_3dz2']+rdf['tbdm_V_3dz2_3dxz']

rdf['tbdm_V_3dz2_2ppi']=rdf['tbdm_V_3dz2_2px']+rdf['tbdm_V_3dz2_2py']
rdf['tbdm_V_3dz2_2p']=rdf['tbdm_V_3dz2_2ppi']+rdf['tbdm_V_3dz2_2pz']

rdf['tbdm_V_2ppi_2pz']=rdf['tbdm_V_2px_2pz']+rdf['tbdm_V_2py_2pz']

rdf['tbdm_V_3d_2pz']=rdf['tbdm_V_3dd_2pz']+rdf['tbdm_V_3dpi_2pz']+rdf['tbdm_V_3dz2_2pz']
rdf['tbdm_V_3d_2ppi']=rdf['tbdm_V_3dd_2ppi']+rdf['tbdm_V_3dpi_2ppi']+rdf['tbdm_V_3dz2_2ppi']
rdf['tbdm_V_3d_2p']=rdf['tbdm_V_3d_2pz']+rdf['tbdm_V_3d_2ppi']
rdf=rdf.drop(columns=[''+\
'tbdm_V_4s_2px',
'tbdm_V_4s_2py',
'tbdm_V_4s_3dxz',
'tbdm_V_4s_3dyz',
'tbdm_V_4s_3dx2y2',
'tbdm_V_4s_3dxy',
'tbdm_V_3dxy_2px',
'tbdm_V_3dxy_2py',
'tbdm_V_3dx2y2_2px',
'tbdm_V_3dx2y2_2py',
'tbdm_V_3dxy_2pz',
'tbdm_V_3dx2y2_2pz',
'tbdm_V_3dxy_3dxz',
'tbdm_V_3dxy_3dyz',
'tbdm_V_3dxz_3dx2y2',
'tbdm_V_3dyz_3dx2y2',
'tbdm_V_3dxy_3dz2',
'tbdm_V_3dz2_3dx2y2',
'tbdm_V_3dyz_2px',
'tbdm_V_3dyz_2py',
'tbdm_V_3dxz_2px',
'tbdm_V_3dxz_2py',
'tbdm_V_3dyz_2pz',
'tbdm_V_3dxz_2pz',
'tbdm_V_3dyz_3dz2',
'tbdm_V_3dz2_3dxz',
'tbdm_V_3dz2_2px',
'tbdm_V_3dz2_2py',
'tbdm_V_2px_2pz',
'tbdm_V_2py_2pz',
'tbdm_V_3dxy_3dx2y2', #These three used in U
'tbdm_V_3dyz_3dxz',
'tbdm_V_2px_2py'])
print('\n'.join(list(rdf)))
print(rdf.shape)

#Pairplot
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(rdf[['energy','tbdm_updown_3d_3d_3d_3d','tbdm_updown_4s_4s_4s_4s','tbdm_updown_2p_2p_2p_2p',
'tbdm_V_4s_3d','tbdm_V_4s_2p','tbdm_V_3d_2p']])
#sns.pairplot(rdf[['energy','tbdm_updown_3d_3d_3d_3d','tbdm_V_3d_2p','obdm_3dz2_3dz2','obdm_2pz_2pz','obdm_4s_4s',
#'obdm_4s_2pz','obdm_3dz2_2pz']])
plt.show()


#OLS
'''
import statsmodels.api as sm
y=rdf['energy']
X=rdf[['obdm_4s_4s', 'tbdm_updown_3d_3d_3d_3d']]
X=sm.add_constant(X)
ols=sm.OLS(y,X).fit()
print(ols.summary())
plt.plot(ols.predict(X),y,'o')
plt.plot(y,y,'--')
plt.show()
exit(0)
'''

'''
#OMP
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import OrthogonalMatchingPursuit
y=rdf['energy']
X=rdf.drop(columns=['energy','energy_err'])
cscores=[]
cscores_err=[]
scores=[]
conds=[]
nparms=[]
for i in range(1,X.shape[1]+1):
#for i in range(41,42):
  print("n_nonzero_coefs="+str(i))
  omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
  omp.fit(X,y)
  nparms.append(i)
  scores.append(omp.score(X,y))
  tmp=cross_val_score(omp,X,y,cv=5)
  cscores.append(tmp.mean())
  cscores_err.append(tmp.std()*2)
  print("R2: ",omp.score(X,y))
  print("R2CV: ",tmp.mean(),"(",tmp.std()*2,")")
  ind=np.abs(omp.coef_)>0
  Xr=X.values[:,ind]
  conds.append(np.linalg.cond(Xr))
  print("Cond: ",np.linalg.cond(Xr))
  print(np.array(list(X))[ind])
  print(omp.coef_[ind])
  '''
'''
  plt.title(fname)
  plt.xlabel("Predicted energy (eV)")
  plt.ylabel("DFT Energy (eV)")
  plt.plot(omp.predict(X),y,'og')
  plt.plot(y,y,'b-')
  plt.savefig(fname.split("p")[0][:-1]+".fit_fix.pdf",bbox_inches='tight')
  #plt.plot(np.arange(len(omp.coef_)),omp.coef_,'o-',label="Nparms= "+str(i))
plt.axhline(0,color='k',linestyle='--')
plt.xticks(np.arange(len(list(X))),list(X),rotation=90)
plt.title(fname)
plt.ylabel("Parameter (eV)")
plt.legend(loc='best')
plt.savefig(fname.split("p")[0][:-1]+".omp_fix.pdf",bbox_inches='tight')
'''
