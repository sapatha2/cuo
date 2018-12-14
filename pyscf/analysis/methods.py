#Generate excitations, energies and rdms for excitations built on CHK state
import numpy as np 
from functools import reduce
from pyscf import lo
import numpy as np
import pandas as pd
from functools import reduce

def gensingles(mo_occ,occ,virt):
  ''' 
  generates a singles excitation list
  input: 
  mo_occ - molecular orbital occupancy of base state
  occ - a list of occupied orbitals in up and down channels (occ[0],occ[1])
  virt - a list of virtual orbitals in up and down channels (occ[0],occ[1])
  output: 
  ex_list - list of mo_occ objects for singles excitations
  '''
  print("-- Singles excitations")
  mo_dm_list=[]
  ex_list=[]
  de_occ=[]
  new_occ=[]
  spin=[]
  #ex_list.append(mo_occ)
  mo_dm_list.append([np.diag(mo_occ[0]),np.diag(mo_occ[1])])
  de_occ.append(0)
  new_occ.append(0)
  spin.append(0)
  for s in [0,1]:
    for j in occ[s]:
      for k in virt[s]:
        tmp=np.array(mo_occ,copy=True)
        assert(tmp[s][j]==1)
        assert(tmp[s][k]==0)
        tmp[s][j]=0
        tmp[s][k]=1
        de_occ.append(j)
        new_occ.append(k)
        #ex_list.append(tmp)
        mo_dm_list.append([np.diag(tmp[0]),np.diag(tmp[1])])
        spin.append(s)
  #return np.array(ex_list),np.array(de_occ),np.array(new_occ),np.array(spin)
  return np.array(mo_dm_list),np.array(de_occ),np.array(new_occ),np.array(spin)

def genex(mf,mo_occ,a,ncore,act,nact,N,Ndet,detgen,c):
  dm_list=[]
  sigu_list=[]
  assert(Ndet>1)
  assert(N>0)
  assert(c<=1.0)

  s=mf.get_ovlp()
  M0=reduce(np.dot,(a.T, s, mf.mo_coeff)) #IAO -> MO for spin up
  
  #Loop states to calculate N+1 states of Ndet+1 determinants
  #First state is always just GS, next are added to GS
  for n in range(N+1):
    #Generate weight object [ CAN BE CHANGED, USING THIS FOR NOWM ... ]
    if(n==0):
      w=np.zeros(Ndet)
      w[0]=1
    else:
      if(c<0): 
        w=np.random.normal(size=Ndet)
        w/=np.sqrt(np.dot(w,w))
      else:
        gauss=np.random.normal(size=Ndet-1)
        gauss/=np.sqrt(np.dot(gauss,gauss))
        w=np.zeros(Ndet)+np.sqrt(c)
        w[1:]=gauss*np.sqrt(1-c)

    #Create det_list object [ CAN BE CHANGED FOR SINGLES, DOUBLES, ... ] 
    det_list=np.zeros((Ndet,2,mo_occ.shape[1]))
    det_list[0]=mo_occ[:,:]
    for i in range(1,Ndet): 
      det_list[i,:,:ncore]=1
      if(detgen=='sd'):
        mydetgen=detgen #Going to hold this until later
        detgen=['s','d'][np.random.randint(2)]
      else:
        mydetgen="NaN"
      
      #Generate determinant through all active space (Very fast)
      if(detgen=='a'):
        det_list[i,0,np.random.choice(act[0],size=nact[0],replace=False)]=1
        det_list[i,1,np.random.choice(act[1],size=nact[1],replace=False)]=1
      #Singles excitations only (A bit slow)
      elif(detgen=='s'):
        det_list[i,:,:]=mo_occ[:,:]
        spin=np.random.randint(2)
        #if(ncore+nact[spin]==act[spin][-1]):
        #  pass
        #else:
        while(ncore+nact[spin]==act[spin][-1]):
          spin=np.random.randint(2)
        q=np.random.randint(low=ncore,high=ncore+nact[spin])
        r=np.random.randint(low=ncore+nact[spin],high=act[spin][-1])
        det_list[i,spin,q]=0
        det_list[i,spin,r]=1
      #Doubles excitations only (Also a bit slow)
      elif(detgen=='d'):
        det_list[i,:,:]=mo_occ[:,:]
        spin=np.random.randint(3)
        if(nact[0]==1 and nact[1]==1): spin=2
        if(spin<2):
          #if(act[spin][-1]-ncore-nact[spin]<=2):
          #  pass
          #else:
          while(act[spin][-1]-ncore-nact[spin]<=2):
            spin=np.random.randint(2)
          q=np.random.choice(np.arange(ncore,ncore+nact[spin]),size=2,replace=False)
          r=np.random.choice(np.arange(ncore+nact[spin],act[spin][-1]),size=2,replace=False)
          det_list[i,spin,q]=0
          det_list[i,spin,r]=1
        else:
          ok=1
          for sp in range(spin):
            if(ncore+nact[sp]==act[sp][-1]):
              spin=np.randon.randint(2)
              ok=0
              i-=1
          if(ok):
            q=np.random.randint(low=ncore,high=ncore+nact[sp])
            r=np.random.randint(low=ncore+nact[sp],high=act[sp][-1])
            det_list[i,sp,q]=0
            det_list[i,sp,r]=1
      else: 
        print(detgen+" not implemented yet")
        exit(0)

      if(mydetgen=='sd'):
        detgen='sd'
    
    #Calculate 1rdm on MO basis 
    dl=np.zeros((det_list.shape[1],det_list.shape[2],det_list.shape[2]))
    dl_v=np.einsum('ijk,i->jk',det_list,w**2)
    dl[0]=np.diag(dl_v[0])
    dl[1]=np.diag(dl_v[1])

    offd=np.einsum('ikl,jkl->kij',det_list,det_list)
    for s in [0,1]:
      for a in range(Ndet):
        for b in range(a,Ndet):
          if(offd[s,a,b]==(ncore+nact[s]-1)): #Check for singles excitation
            ind=np.where((det_list[a,s,:]-det_list[b,s,:])!=0)[0]
            M=np.zeros(dl[s].shape)
            M[ind[0],ind[1]]=1
            M[ind[1],ind[0]]=1
            dl[s]+=w[a]*w[b]*M
  
    #Calculate Us
    cncm=np.einsum('isn,jsm->ijsnm',det_list,det_list)
    tmp=np.einsum('ijsnm,skm->ijsnk',cncm,np.array([M0,M0]))
    MU=np.einsum('ijsnk,skn->ksij',tmp,np.array([M0,M0]))
        
    sigu=np.einsum('ksij,i->ksj',MU,w)
    sigu=np.einsum('ksj,j->ks',sigu,w)
    
    sigu_list.append(sigu[:,0]*sigu[:,1])
    dm_list.append(dl)

  dm_list=np.array(dm_list)
  sigu_list=np.array(sigu_list)
  return dm_list,sigu_list
