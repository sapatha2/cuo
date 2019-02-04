import numpy as np 
from functools import reduce
from scipy.optimize import minimize 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import normalize

def tmat():
  return np.array([[0,0,1,1],[0,0,-1,-1],[1,-1,0,0],[1,-1,0,0]])

def Umat(): 
  return np.diag([0,0,1,1])

def Hmat(t,U): 
 return t*tmat() + U*Umat()

def parmvec(H,v):
  #Returns parameter object 
  w,vr=np.linalg.eigh(H)
  v=np.dot(v,vr.T)
  r=np.zeros((v.shape[0],2))
  r[:,0]=np.sum(v*np.dot(v,tmat()),axis=1)
  r[:,1]=np.sum(v*np.dot(v,Umat()),axis=1)
  return r

def energy(H,v):
  r=parmvec(H,v)
  ret=0
  for i in range(len(r)):
    for j in range(i+1,len(r)):
      ret+=1./np.sqrt(np.dot(r[i]-r[j],r[i]-r[j]))
  print("Energy :",ret)
  return ret

#Monte carlo loop
def MC(H,gsw=0.9,nsample=10,nstep=100,tstep=1,beta=1):
  v=np.random.normal(size=(nsample,3)) #Don't include GS coefficient
  v=normalize(v,norm='l2')*np.sqrt(1-gsw)

  for step in range(nstep):
    vnew=v+tstep*np.random.normal(size=(nsample,3)) #Kinetic energy 
    vnew=normalize(vnew,norm='l2')*np.sqrt(1-gsw)
    
    v=np.concatenate((np.sqrt(gsw)*np.ones((nsample,1)),v),axis=1)
    vnew=np.concatenate((np.sqrt(gsw)*np.ones((nsample,1)),vnew),axis=1)
    Eold=energy(H,v)
    Enew=energy(H,vnew)

    accept=np.exp(-beta*(Enew-Eold))>np.random.rand(nsample)
    v[accept]=vnew[accept]
    v=v[:,1:] #Don't include GS coefficient
  
  v=np.concatenate((np.sqrt(gsw)*np.ones((nsample,1)),v),axis=1)
  print(v)

t=-1
U=4
H=Hmat(t,U)
MC(H,gsw=0.9,tstep=0.1,beta=3)
