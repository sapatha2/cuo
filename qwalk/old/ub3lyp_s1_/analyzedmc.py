import os 
import numpy as np 
import scipy as sp 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.linear_model
from sklearn.metrics import r2_score
from ed import ED
from pyscf import gto, scf, ao2mo, cc, fci, mcscf, lib
from pyscf.scf import ROKS
from functools import reduce 
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.optimize import linear_sum_assignment 
from find_connect import  *
import matplotlib as mpl 
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import itertools

######################################################################################
#FROZEN METHODS
#Collect df
def collect_df():
  df=None
  for basestate in range(11):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)

      small_df['basestate']=basestate
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=0.5
      if(df is None): df = small_df
      else: df = pd.concat((df,small_df),axis=0,sort=True)
  
  for basestate in range(6):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../ub3lyp_s3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+11
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)

  for basestate in range(4):
    for gsw in [0.2,0.4,0.6,0.8,1.0]:
      f='../../ub3lyp_extra_1/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+17
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
 
  for basestate in range(2):
    for gsw in np.arange(-1.0,1.2,0.2):
      f='../../ub3lyp_extra_3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+21
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  
  #formatted_gosling_2
  '''
  for basestate in range(4):
    for gsw in np.arange(0.2,1.2,0.2):
      f='../../ub3lyp_extra_extra/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+23
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  '''
  
  #formatted_gosling_3
  '''
  for basestate in range(11,13):
    for gsw in np.arange(0.2,1.2,0.2):
      f='../../ub3lyp_1/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+23
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=0.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  
  for basestate in [6]:
    for gsw in np.arange(0.2,1.2,0.2):
      f='../../ub3lyp_3/gsw'+str(np.round(gsw,2))+'b'+str(basestate)+'/dmc_gosling.pickle' 
      small_df=pd.read_pickle(f)
    
      small_df['basestate']=basestate+35
      if(gsw==1.0): small_df['basestate']=-small_df['basestate']
      small_df['Sz']=1.5
      df = pd.concat((df,small_df),axis=0,sort=True)
  '''
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

  df['Jd']=np.zeros(df.shape[0])
  orb1=[1,1,1,1,2,2,2,3,3,4]
  orb2=[2,3,4,5,3,4,5,4,5,5]
  for i in range(len(orb1)):
    df['Jd']+=df['j_'+str(orb1[i])+'_'+str(orb2[i])]
  df['Jsd']=df['j_0_1']+df['j_0_2']+df['j_0_3']+df['j_0_4']+df['j_0_5']
  df['Jcu']=df['Jsd']+df['Jd']
  return df #format_df_iao(df)

#Get IAO psums
def format_df_iao(df):
  df['beta']=-1000
  #LOAD IN IAOS
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]

  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()

  #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s']
  #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s

  df['iao_n_3dd']=0
  df['iao_n_3dpi']=0
  df['iao_n_3dz2']=0
  df['iao_n_3d']=0
  df['iao_n_2pz']=0
  df['iao_n_2ppi']=0
  df['iao_n_4s']=0
  df['iao_t_pi']=0
  df['iao_t_dz']=0
  df['iao_t_ds']=0
  df['iao_t_sz']=0
  for z in range(df.shape[0]):
    print(z)
    e=np.zeros((9,9))
    orb1=[1,2,3,4,5,6,7,8,9,1,2,3,8,3]
    orb2=[1,2,3,4,5,6,7,8,9,6,7,8,9,9]
    for i in range(len(orb1)):
      e[orb1[i]-1,orb2[i]-1]=df['mo_'+str(orb1[i])+'_'+str(orb2[i])].values[z]
    mo_to_iao = reduce(np.dot,(mo.T,s,iao))
    e = reduce(np.dot,(mo_to_iao.T,e,mo_to_iao))
    e[np.abs(e)<1e-10]=0
    e=(e+e.T)/2
    df['iao_n_3dd'].iloc[z]=np.sum(np.diag(e)[[0,1]])
    df['iao_n_3dpi'].iloc[z]=np.sum(np.diag(e)[[2,3]])
    df['iao_n_3dz2'].iloc[z]=np.diag(e)[6]
    df['iao_n_3d'].iloc[z]=np.sum(np.diag(e)[[0,1,2,3,6]])
    df['iao_n_2pz'].iloc[z]=np.sum(np.diag(e)[7])
    df['iao_n_2ppi'].iloc[z]=np.sum(np.diag(e)[[4,5]])
    df['iao_n_4s'].iloc[z]=np.sum(np.diag(e)[8])
    df['iao_t_pi'].iloc[z]=2*(e[3,4]+e[2,5])
    df['iao_t_ds'].iloc[z]=2*e[6,8]
    df['iao_t_dz'].iloc[z]=2*e[6,7]
    df['iao_t_sz'].iloc[z]=2*e[7,8]
  return df

######################################################################################
#ANALYSIS CODE 

#LASSO 
def lasso(df,max_model,alphas=np.arange(1e-6,0.105,0.002)):
  '''
  input: 
    df - parameters and energies
    max_model - the maximum largest model to include
    alphas - importance of L1 term in Lasso
  output: 
    sel_model - selected model
    nz - number of non zero parameters
  '''
  sel_model = []
  nz = []
  X = df[max_model]
  y = df['energy']
  for alpha in alphas:
    lasso = sklearn.linear_model.Lasso(alpha=alpha,fit_intercept=True,selection='random')
    lasso = lasso.fit(X,y)
    ind = tuple(lasso.coef_.nonzero()[0])
    sel_model.append(ind)
    nz.append(len(ind))
  return np.array(sel_model), np.array(nz)

def oneparm_valid(df,max_model,sel_model,nbs=20):
  oneparm_df = None
  #Loop model
  for model in sel_model:
    params = []
    r2 = []
    #Loop bootstrap sample
    for n in range(nbs): 
      #Bootstrap sample data frame
      dfn = df.sample(n=df.shape[0],replace=True)
      X = dfn[np.array(max_model)[list(model)]]
      X['const'] = 1
      y = dfn['energy']

      #Fit regression
      ols = sm.OLS(y,X).fit()
      params_ = np.zeros(len(max_model))
      params_[list(model)] = ols.params[:-1]
      params.append(params_)
      r2.append(r2_score(ols.predict(),y))

    params_mu = np.mean(params,axis=0)
    params_err = np.std(params,axis=0)
    r2_mu = np.mean(r2,axis=0)
    r2_err = np.std(r2,axis=0)
    
    data = np.array([r2_mu, r2_err] + list(params_mu) + list(params_err))[np.newaxis,:]
    columns = ['r2_mu','r2_err'] + list(max_model) + [x+'_err' for x in max_model]
    d = pd.DataFrame(data = data, columns = columns)
    if(oneparm_df is None): oneparm_df = d
    else: oneparm_df = pd.concat((oneparm_df,d),axis=0)
  
  return oneparm_df

def exact_diag(df,max_model,sel_model,nbs=20):
  eig_df = None
  #Loop model
  z=-1
  for model in sel_model:
    z+=1
    #Loop bootstrap sample
    for n in range(nbs): 
      #Bootstrap sample data frame
      dfn = df.sample(n=df.shape[0],replace=True)
      X = dfn[np.array(max_model)[list(model)]]
      X['const'] = 1
      y = dfn['energy']

      #Fit regression
      ols = sm.OLS(y,X).fit()
      params = np.zeros(len(max_model))
      params[list(model)] = ols.params[:-1]
      
      #Do ED
      norb=9
      nelec=(8,7)
      nroots=30
      res1=ED(params,nroots,norb,nelec)

      nelec=(9,6)
      nroots=30
      res3=ED(params,nroots,norb,nelec)
    
      E = res1[0]
      Sz = np.ones(len(E))*0.5
      ci=np.array(res1[1])
      ci=np.reshape(ci,(ci.shape[0],ci.shape[1]*ci.shape[2]))
      d = pd.DataFrame({'energy':E,'Sz':Sz})
      d['ci']=list(ci)

      E = res3[0]
      Sz = np.ones(len(E))*1.5
      ci=np.array(res3[1])
      ci=np.reshape(ci,(ci.shape[0],ci.shape[1]*ci.shape[2]))
      d2 = pd.DataFrame({'energy':E,'Sz':Sz})
      d2['ci']=list(ci)

      d=pd.concat((d,d2),axis=0)
      d['model'] = z
      d['bs_index'] = n
      d['energy'] += ols.params[-1]

      #Concatenate
      if(eig_df is None): eig_df = d
      else: eig_df = pd.concat((eig_df,d),axis=0)

  return eig_df

def sort_ed(df):
  print("SORT_ED")
  #Sorting
  sorted_df = None
  for model in range(max(df['model'])+1):
    d = df[(df['model']==model)]
    for j in range(max(d['bs_index'])+1):
      offset=0
      for Sz in [0.5,1.5]:
        a = d[(d['bs_index']==0)&(d['Sz']==Sz)]
        amat = np.array(list(a['ci']))

        b = d[(d['bs_index']==j)&(d['Sz']==Sz)]
        bmat = np.array(list(b['ci']))

        #Apply permutation to pair up non degenerate states
        cost = -1.*np.dot(amat,bmat.T)**2
        row_ind, col_ind = linear_sum_assignment(cost)
        bmat = bmat[col_ind,:]

        #Gotta do some extra work for the degenerate states
        #Get connected groups
        abdot = np.dot(amat,bmat.T)
        mask = (abdot**2 > 1e-5)
        connected_sets, home_nodes = find_connected_sets(mask)
       
        #Loop connected groups to see which ones correspond to degenerate states
        for i in connected_sets:
          len_check = False
          sub_ind = None
          sum_check = False
          eig_check = False
          #Check length as criteria for degeneracy
          if(len(connected_sets[i])>1):
            sub_ind=list(connected_sets[i])
            len_check = True

          if(len_check):
            degen_a = len(set(np.around(a['energy'].iloc[row_ind[sub_ind]],6)))
            degen_b = len(set(np.around(b['energy'].iloc[col_ind[sub_ind]],6)))
            if((degen_a == 1)&(degen_b ==1)): eig_check = True

          #Check that the degenerate space is actually spanned properly
          if(eig_check):
            sub_mat = abdot[sub_ind][:,sub_ind]
            bmat[row_ind[sub_ind],:]=np.dot(sub_mat,bmat[row_ind[sub_ind],:]) 
        
        #We finally have bmat ordered correctly and everything 
        dtmp = pd.DataFrame({'energy':b['energy'].values[col_ind],'Sz':b['Sz'].values,'bs_index':b['bs_index'].values})
        dtmp['ci']=list(bmat)
        dtmp['eig']=np.arange(dtmp.shape[0]) + offset
        dtmp['model']=b['model'].values
        offset += dtmp.shape[0]
        if(sorted_df is None): sorted_df = dtmp
        else: sorted_df = pd.concat((sorted_df,dtmp),axis=0)
  return sorted_df

def desc_ed(df):
  print("DESC_ED")
  #Get descriptors from ci
  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)
  sigUs = []
  sigJsd = []
  sigNdz2 = []
  sigNdpi = []
  sigNdd =[]
  sigNd = []
  sigN2pz = []
  sigN2ppi = []
  sigN4s = []
  sigTpi = []
  sigTds = []
  sigTdz = []
  sigTsz = []

  sigMONdz2 = []
  sigMONdpi = []
  sigMONdd =[]
  sigMONd = []
  sigMON2pz = []
  sigMON2ppi = []
  sigMON4s = []
  sigMOTpi = []
  sigMOTds = []
  sigMOTdz = []
  sigMOTsz = []

  #GET MO TO IAO MATRIX
  act_iao=[5,9,6,8,11,12,7,13,1]
  iao=np.load('../../../pyscf/ub3lyp_full/b3lyp_iao_b.pickle')
  iao=iao[:,act_iao]
  
  #LOAD IN MOS
  act_mo=[5,6,7,8,9,10,11,12,13]
  chkfile='../../../pyscf/chk/Cuvtz_r1.725_s1_B3LYP_1.chk'
  mol=lib.chkfile.load_mol(chkfile)
  m=ROKS(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
  mo=m.mo_coeff[:,act_mo]
  s=m.get_ovlp()
  mo_to_iao = reduce(np.dot,(mo.T,s,iao))

  for i in range(df.shape[0]):
    ci = df['ci'].iloc[i]
    norb=9
    nelec=(8,7)
    if(df['Sz'].iloc[i]==1.5): nelec=(9,6)
    ci = ci.reshape((sp.misc.comb(norb,nelec[0],exact=True),sp.misc.comb(norb,nelec[1],exact=True)))
    dm2=cis.make_rdm12s(ci,norb,nelec)

    sigUs.append(dm2[1][1][8,8,8,8])
    
    Jsd = 0
    for j in [0,1,2,3,6]:
      Jsd += 0.25*(dm2[1][0][8,8,j,j] + dm2[1][2][8,8,j,j] - dm2[1][1][8,8,j,j] - dm2[1][1][j,j,8,8])-\
             0.5*(dm2[1][1][j,8,8,j] + dm2[1][1][8,j,j,8]) 
    sigJsd.append(Jsd)
  
    dm = dm2[0][0] + dm2[0][1]
   
    #IAO ordering: ['del','del','yz','xz','x','y','z2','z','s'] 
    sigNdz2.append(dm[6,6])
    sigNdpi.append(dm[2,2]+dm[3,3])
    sigNdd.append(dm[0,0]+dm[1,1])
    sigNd.append(dm[0,0]+dm[1,1]+dm[2,2]+dm[3,3]+dm[6,6])
    sigN2pz.append(dm[7,7])
    sigN2ppi.append(dm[4,4]+dm[5,5])
    sigN4s.append(dm[8,8])
    sigTpi.append(2*(dm[3,4]+dm[2,5]))
    sigTds.append(2*dm[6,8])
    sigTdz.append(2*dm[6,7])
    sigTsz.append(2*dm[7,8])
  
    #MO ordering:  dxz, dyz, dz2, delta, delta, px, py, pz, 4s
    mo_dm = reduce(np.dot,(mo_to_iao,dm,mo_to_iao.T))
    sigMONdz2.append(mo_dm[2,2])
    sigMONdpi.append(mo_dm[0,0]+mo_dm[1,1])
    sigMONdd.append(mo_dm[3,3]+mo_dm[4,4])
    sigMONd.append(mo_dm[0,0]+mo_dm[1,1]+mo_dm[2,2]+mo_dm[3,3]+mo_dm[4,4])
    sigMON2pz.append(mo_dm[7,7])
    sigMON2ppi.append(mo_dm[6,6]+mo_dm[5,5])
    sigMON4s.append(mo_dm[8,8])
    sigMOTpi.append(2*(mo_dm[0,5]+mo_dm[1,6]))
    sigMOTds.append(2*mo_dm[2,8])
    sigMOTdz.append(2*mo_dm[2,7])
    sigMOTsz.append(2*mo_dm[7,8])

  df['iao_n_3dz2']=sigNdz2
  df['iao_n_3dpi']=sigNdpi
  df['iao_n_3dd']=sigNdd
  df['iao_n_3d']=sigNd
  df['iao_n_2pz']=sigN2pz
  df['iao_n_2ppi']=sigN2ppi
  df['iao_n_4s']=sigN4s
  df['iao_t_pi']=sigTpi
  df['iao_t_ds']=sigTds
  df['iao_t_dz']=sigTdz
  df['iao_t_sz']=sigTsz

  df['mo_n_3dz2']=sigMONdz2
  df['mo_n_3dpi']=sigMONdpi
  df['mo_n_3dd']=sigMONdd
  df['mo_n_3d']=sigMONd
  df['mo_n_2pz']=sigMON2pz
  df['mo_n_2ppi']=sigMON2ppi
  df['mo_n_4s']=sigMON4s
  df['mo_t_pi']=sigMOTpi
  df['mo_t_ds']=sigMOTds
  df['mo_t_dz']=sigMOTdz
  df['mo_t_sz']=sigMOTsz

  df['Us']=sigUs
  df['Jsd']=sigJsd  
  return df

def avg_ed(df):
  print("AVG_ED")
  avg_df = None
  for model in range(max(df['model'])+1):
    for eig in range(max(df['eig'])+1):
      sub_df = df[(df['model']==model) & (df['eig']==eig)]
      data = sub_df.values
      means = np.mean(data,axis=0)
      u = np.percentile(data,97.5,axis=0) - means
      l = means - np.percentile(data,2.5,axis=0)

      d=pd.DataFrame(data=np.array(list(means) + list(u)+list(l))[:,np.newaxis].T,
      columns=list(sub_df) + [x+'_u' for x in list(sub_df)] + [x+'_l' for x in list(sub_df)])

      if(avg_df is None): avg_df = d
      else: avg_df = pd.concat((avg_df,d),axis=0)
  return avg_df

#Plot eigenvalues and eigenproperties
def plot_ed(full_df,av_df,model,save=True):
  norm = mpl.colors.Normalize(vmin=0, vmax=3.75)
  limits = [(0.5,2.5),(2.5,4.5),(1.5,4.5),(0.5,2.5),(1.5,4.5),(0,1.5),
  (-1.0,1.5),(-1.5,1.5),(-1.5,1.5),(-1.5,1.5),(-1,1),(-0.5,1.0)]
    
  rgba_color = plt.cm.Blues(norm(1.75))
  rgba_color2 = plt.cm.Oranges(norm(1.75))
  z=-1 
  fig, axes = plt.subplots(nrows=2,ncols=6,sharey=True,figsize=(12,6))
  for parm in ['iao_n_3dz2','iao_n_3dpi','iao_n_3dd','iao_n_2pz','iao_n_2ppi','iao_n_4s',
  'iao_t_pi','iao_t_ds','iao_t_dz','iao_t_sz','Jsd','Us']:
    z+=1 
    ax = axes[z//6,z%6]

    #DMC Data
    full_df['energy'] -= min(full_df['energy'])

    f_df = full_df[full_df['Sz']==0.5]
    x = f_df[parm].values
    y = f_df['energy'].values
    yerr = f_df['energy_err'].values
    ax.errorbar(x,y,yerr,fmt='s',c=rgba_color,alpha=0.5)

    f_df = full_df[full_df['Sz']==1.5]
    x = f_df[parm].values
    y = f_df['energy'].values
    yerr = f_df['energy_err'].values
    ax.errorbar(x,y,yerr,fmt='s',c=rgba_color2,alpha=0.5)

    #Eigenstates
    minE = min(av_df[av_df['model']==model]['energy'])
    sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==0.5)]
    sub_df['energy'] -= minE
    x=sub_df[parm].values
    xerr_u=sub_df[parm+'_u'].values
    xerr_d=sub_df[parm+'_l'].values
    y=sub_df['energy'].values
    yerr_u=sub_df['energy_u'].values
    yerr_d=sub_df['energy_l'].values
    ax.errorbar(x,y,xerr=[xerr_d,xerr_u],yerr=[yerr_d,yerr_u],markeredgecolor='k',fmt='o',c=rgba_color)

    sub_df = av_df[(av_df['model']==model)&(av_df['Sz']==1.5)]
    sub_df['energy'] -= minE
    x=sub_df[parm].values
    xerr_u=sub_df[parm+'_u'].values
    xerr_d=sub_df[parm+'_l'].values
    y=sub_df['energy'].values
    yerr_u=sub_df['energy_u'].values
    yerr_d=sub_df['energy_l'].values
    ax.errorbar(x,y,xerr=[xerr_d,xerr_u],yerr=[yerr_d,yerr_u],markeredgecolor='k',fmt='o',c=rgba_color2)

    ax.axhline(min(full_df['energy'])+3,ls='--',c='k')
    ax.set_xlabel(parm)
    ax.set_ylabel('energy (eV)')
    ax.set_xlim(limits[z])
    ax.set_ylim((-0.2,4.5))
  plt.suptitle('Ed model '+str(model))
  plt.show()
  return -1

######################################################################################
#RUN
def analyze(df=None,save=False):
  #Analysis (Collect model information)
  '''
  max_model = ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz','mo_t_sz','mo_t_ds','Jsd','Us']
  core = [0,1,2,8]
  hopping=[3,4,5,6,7]
  sel_model=[core]
  for n in range(1,len(hopping)+1):
    s=set(list(hopping))
    models=list(map(set,itertools.combinations(s,n)))
    sel_model+=[core+list(m) for m in models]
  print(len(sel_model))

  #Parameters and regression 
  oneparm_df = oneparm_valid(df,max_model,sel_model,nbs=100)

  #Exact diagonalization of the models
  eig_df = exact_diag(df,max_model,sel_model,nbs=100)

  #Sorting and averaging 
  eig_df = sort_ed(eig_df)
  eig_df = desc_ed(eig_df)
  avg_eig_df = avg_ed(eig_df.drop(columns=['ci']))

  oneparm_df.to_pickle('analysis/oneparm.pickle')
  eig_df.to_pickle('analysis/eig.pickle')
  avg_eig_df.to_pickle('analysis/avg_eig.pickle')
  exit(0)
  '''

  eig_df = pd.read_pickle('analysis/eig.pickle')
  eig_df = desc_ed(eig_df)
  avg_eig_df = avg_ed(eig_df.drop(columns=['ci']))
  avg_eig_df.to_pickle('analysis/avg_eig.pickle')
  exit(0)

  #Select and plot unique models
  '''
  unique_models = []
  oneparm_df = pd.read_pickle('analysis/oneparm.pickle').drop(columns=['r2_mu','r2_err'])
  for model in np.arange(32):
    non_zero = np.nonzero(oneparm_df.iloc[model])[0]
    half = int(len(non_zero)/2)
    vals = oneparm_df.iloc[model].values[non_zero[:half]]
    errs = oneparm_df.iloc[model].values[non_zero[half:]]
    
    u = vals + 2*errs
    l = vals - 2*errs
    sign = np.prod(np.sign(u*l))
    if(sign > 0): 
      if(oneparm_df.iloc[model]['Jsd']!=0):
        unique_models.append(model)
  print('unique models: ',unique_models)

  avg_eig_df = pd.read_pickle('analysis/avg_eig.pickle')
  for model in unique_models:
    plot_ed(df,avg_eig_df,model=model)
  exit(0)
  '''

if __name__=='__main__':
  #DATA COLLECTION
  #df=collect_df()
  #df=format_df(df)
  #df=format_df_iao(df)
  #df.to_pickle('formatted_gosling.pickle')
  
  df = pd.read_pickle('formatted_gosling.pickle')
  analyze(df)
