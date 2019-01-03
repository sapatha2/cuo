#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
from functools import reduce
from pyscf import lo
import numpy as np
import pandas as pd
from functools import reduce

def genex(mo_occ,mo_energy,ncore,act,nact,N,Ndet,detgen,c,beta):
  dm_list=[]
  sigu_list=[]
  assert(Ndet>1)
  assert(N>0)
  assert(c<=1.0)
  
  #Loop states to calculate N+1 states of Ndet+1 determinants
  #First state is always just GS, next are added to GS
  for n in range(N+1):

    #Create det_list object [ CAN BE CHANGED FOR SINGLES, DOUBLES, ... ] 
    det_list=np.zeros((Ndet,2,mo_occ.shape[1]))
    det_list[0]=mo_occ.copy() #mo_occ[:,:]
    for i in range(1,Ndet): 
      #det_list[i,:,:ncore]=1
      det_list[i,:,ncore]=1
      #Generate determinant through all active space (Very fast)
      if(detgen=='a'):
        det_list[i,0,np.random.choice(act[0],size=nact[0],replace=False)]=1
        det_list[i,1,np.random.choice(act[1],size=nact[1],replace=False)]=1
      #Singles excitations only (A bit slow)
      elif(detgen=='s'):
        det_list[i,:,:]=mo_occ.copy() #mo_occ[:,:]
        spin=np.random.randint(2)
        while(ncore+nact[spin]==act[spin][-1]+1):
          spin=np.random.randint(2)
        q=np.random.randint(low=ncore,high=ncore+nact[spin])
        r=np.random.randint(low=ncore+nact[spin],high=act[spin][-1]+1)
        det_list[i,spin,q]=0
        det_list[i,spin,r]=1
      else: 
        print(detgen+" not implemented yet")
        exit(0)

    #Generate weight object [ CAN BE CHANGED, USING THIS FOR NOW ... ]
    dete=np.einsum('k,isk->i',mo_energy,det_list)*27.2114
    dete-=dete[0]
    dete/=2
    assert(np.sum(dete)>=0)
    if(n==0):
      w=np.zeros(Ndet)
      w[0]=1
    else:
      gauss=np.random.normal(size=Ndet-1)
      gauss*=np.exp(-1*beta*dete[1:])
      gauss/=np.sqrt(np.dot(gauss,gauss))
      w=np.zeros(Ndet)+np.sqrt(c)
      w[1:]=gauss*np.sqrt(1-c)

    #Calculate 1rdm on MO basis 
    dl=np.zeros((det_list.shape[1],det_list.shape[2],det_list.shape[2]))
    dl_v=np.einsum('ijk,i->jk',det_list,w**2)
    dl[0]=np.diag(dl_v[0])
    dl[1]=np.diag(dl_v[1])

    offd=np.einsum('ikl,jkl->kij',det_list,det_list)
    for s in [0,1]:
      for a in range(Ndet):
        for b in range(a+1,Ndet):
          sflip=np.mod(s+1,2)
          if((offd[s,a,b]==(sum(ncore)+nact[s]-1)) and (offd[sflip,a,b]==(sum(ncore)+nact[sflip]))): #Check for singles excitation
            ind=np.where((det_list[a,s,:]-det_list[b,s,:])!=0)[0]
            M=np.zeros(dl[s].shape)
            M[ind[0],ind[1]]=1
            M[ind[1],ind[0]]=1
            dl[s]+=w[a]*w[b]*M

    dm_list.append(dl)
  dm_list=np.array(dm_list)
  return dm_list
