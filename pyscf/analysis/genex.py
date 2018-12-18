import numpy as np 
import matplotlib.pyplot as plt
from methods import gensingles,genex
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
from functools import reduce
import pandas as pd 
import seaborn as sns 
import statsmodels.api as sm 
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import cross_val_score

f='b3lyp_iao_b.pickle'
a=np.load(f)

r=1.963925
method='B3LYP'
basis='vtz'
el='Cu'
charge=0

escf={'-1':-213.5799904,
       '1':-213.5799904,
       '3':-213.522801}
occ={'-1': [np.arange(6,14)-1,np.arange(6,13)-1], 
      '1': [np.arange(6,14)-1,np.arange(6,13)-1],
      '3': [np.arange(6,15)-1,np.arange(6,12)-1]}
virt={'-1': [np.arange(14,15)-1,np.arange(13,15)-1], 
       '1': [np.arange(14,15)-1,np.arange(13,15)-1], 
       '3': [np.arange(15,15)-1,np.arange(12,15)-1]}
ncore={'-1':5,
        '1':5,
        '3':5}
nact={'-1':[8,7],
       '1':[8,7],
       '3':[9,6]}
act={'-1':[np.arange(5,15),np.arange(5,15)],
      '1':[np.arange(5,15),np.arange(5,15)],
      '3':[np.arange(5,15),np.arange(5,15)]}

data=None
for mol_spin in [-1,1,3]:
  chkfile="../chkfiles/"+el+basis+"_r"+str(r)+"_c"+str(charge)+"_s"+str(mol_spin)+"_"+method+".chk"
  mol=lib.chkfile.load_mol(chkfile)

  if("U" in method): m=UHF(mol)
  else: m=ROHF(mol)
  m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))

  ##################
  #BUILD EXCITATIONS 
  mo_occ=np.array([np.ceil(m.mo_occ-m.mo_occ/2),np.floor(m.mo_occ/2)])
  detgen='s'
  N=200
  Ndet=20
  c=0.9
  st=str(mol_spin)
  dm_list,u=genex(m,mo_occ,a,ncore[st],act[st],nact[st],N,Ndet,detgen,c)

  #IAO rdms
  s=m.get_ovlp()
  M=m.mo_coeff
  M=reduce(np.dot,(a.T,s,M))
  iao_dm_list=np.einsum('ijkl,mk->ijml',dm_list,M)
  iao_dm_list=np.einsum('ijml,nl->ijmn',iao_dm_list,M)
  
  ##################
  #Traces
  '''
  plt.subplot(211)
  plt.title("MO trace singles, S="+str(mol_spin)+", IAO="+f)
  tr_mo=np.einsum('ijmm->ij',dm_list)
  plt.plot(tr_mo[:,0],'*')
  plt.plot(tr_mo[:,1],'s')
  
  plt.subplot(212)
  plt.title("IAO trace singles, S="+str(mol_spin)+", IAO="+f)
  tr=np.einsum('ijmm->ij',iao_dm_list)
  plt.plot(tr[:,0],'*')
  plt.plot(tr[:,1],'s')
 
  plt.savefig(f.split(".")[0]+'_s'+str(mol_spin)+'_tr.pdf',bbox_inches='tight')
  plt.close()
  '''
  
  e=np.einsum('ijll,l->ij',dm_list,m.mo_energy)
  e=e[:,0]+e[:,1]
  e-=e[0] #Eigenvalue difference
  e+=escf[str(mol_spin)] #Base SCF difference

  #Number occupations 
  n=np.einsum('ijmm->ijm',iao_dm_list)
  labels=np.array(["3s","4s","3px","3py","3pz","3dxy","3dyz","3dz2","3dxz","3dx2y2","2s","2px","2py","2pz"])
  rel=[1,5,6,7,8,9,11,12,13]
  n=n[:,0,rel]+n[:,1,rel]
  
  #Hopping 
  trel=np.array([[1,1,6,8,7],[7,13,12,11,13]])
  tlabels=np.array([labels[trel[0]],labels[trel[1]]]).T
  tlabels=[x[0]+"-"+x[1] for x in tlabels]
  t=iao_dm_list[:,0,trel[0],trel[1]]+iao_dm_list[:,0,trel[1],trel[0]]+\
    iao_dm_list[:,1,trel[0],trel[1]]+iao_dm_list[:,1,trel[1],trel[0]] #Hermitian conjugates required

  #U
  ulabels=list(labels)
  ulabels=np.array([x+'_u' for x in ulabels])
  u=u[:,rel]

  #Data object
  d=np.concatenate((e[:,np.newaxis],n,t,u),axis=1)
  if(data is None): data=d
  else: data=np.concatenate((data,d),axis=0)

#Full data frame
df=pd.DataFrame(data,columns=["E"]+list(labels[rel])+list(tlabels)+list(ulabels[rel]))
df['E']-=df['E'][0]
df['E']*=27.2114
df['3dd']=df['3dxy']+df['3dx2y2']
df['3dpi']=df['3dxz']+df['3dyz']
df['2ppi']=df['2px']+df['2py']
df['tpi']=df['3dyz-2py']+df['3dxz-2px']
df['3dd_u']=df['3dxy_u']+df['3dx2y2_u']
df['3dpi_u']=df['3dxz_u']+df['3dyz_u']
df['2ppi_u']=df['2px_u']+df['2py_u']
df['3dz2-2pz']*=-1  #Get the sign structure correct
df['4s-2pz']*=-1    #Get the sign structure correct
df=df.drop(columns=['3dxz','3dyz','2px','2py','3dyz-2py','3dxz-2px','3dx2y2','3dxy',
'3dxz_u','3dyz_u','2px_u','2py_u','3dx2y2_u','3dxy_u'])

#Dump 
df.to_pickle(f.split(".")[0]+"_"+str(detgen)+"_N"+str(N)+"_Ndet"+str(Ndet)+"_c"+str(c)+".pickle")
