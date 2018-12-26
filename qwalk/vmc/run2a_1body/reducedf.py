import pandas as pd 
import numpy as np 

fname='a_Ndet10_gsw0.8_df.pickle'
df=pd.read_pickle(fname)

#Correct labels
labels=np.array(["4s","3dxy","3dyz","3dz2","3dxz","3dx2y2","2px","2py","2pz"])
iaos=  np.array([ 2,   6,     7,     8,     9,     10,      12,   13,   14])
fix_cols=[]
for col in list(df):
  l=col.split("_")
  if(len(l)>=4):
    a=int(col.split("_")[2])
    b=int(col.split("_")[3])
    i=np.where(iaos==a)[0][0]
    j=np.where(iaos==b)[0][0]
    fix_col=col.split("_")
    fix_col[2]=labels[i]
    fix_col[3]=labels[j]
    fix_col='_'.join(fix_col)
  else: fix_col=col
  fix_cols.append(fix_col)
df.columns=fix_cols

#Energy units
df['energy']*=27.2114 #eV
df['energy_err']*=27.2114

#Sum over spins
nu=['obdm_up_'+i+'_'+j for i in labels for j in labels]
nd=['obdm_down_'+i+'_'+j for i in labels for j in labels]
ntot=['obdm_'+i+'_'+j for i in labels for j in labels]
n=df[nu].values+df[nd].values
df[ntot]=pd.DataFrame(n,index=df.index)
df=df.drop(columns=nd+nu)

nu=[x+'_err' for x in nu]
nd=[x+'_err' for x in nd]
ntot=[x+'_err' for x in ntot]
n=np.sqrt(df[nu].values**2 + df[nd].values**2)
df[ntot]=pd.DataFrame(n,index=df.index)
df=df.drop(columns=nd+nu)

#Reduced DF
n=['obdm_'+str(i)+'_'+str(i) for i in labels]
t=['obdm_4s_2pz','obdm_4s_3dz2','obdm_3dyz_2py','obdm_3dz2_2pz','obdm_3dxz_2px']
rdf=df[['energy']+['energy_err']+n+t]
rdf['obdm_3dpi_3dpi']=rdf['obdm_3dxz_3dxz']+rdf['obdm_3dyz_3dyz']
rdf['obdm_3dd_3dd']=rdf['obdm_3dxy_3dxy']+rdf['obdm_3dx2y2_3dx2y2']
rdf['obdm_2ppi_2ppi']=rdf['obdm_2px_2px']+rdf['obdm_2py_2py']
rdf['obdm_3dpi_2ppi']=rdf['obdm_3dxz_2px']+rdf['obdm_3dyz_2py']
rdf['obdm_3dz2_2pz']*=-1  #Get the sign structure correct
rdf['obdm_4s_2pz']*=-1    #Get the sign structure correct
rdf=rdf.drop(columns=['obdm_3dxz_3dxz','obdm_3dyz_3dyz',
                      'obdm_3dxy_3dxy','obdm_3dx2y2_3dx2y2','obdm_2px_2px',
                      'obdm_2py_2py','obdm_3dxz_2px','obdm_3dyz_2py'])
rdf[['obdm_4s_2pz','obdm_4s_3dz2','obdm_3dz2_2pz','obdm_3dpi_2ppi']]*=2 #Hermitian conjugate
rdf.to_pickle(fname+'R')
