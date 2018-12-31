import numpy as np 
import matplotlib.pyplot as plt
fname='gs2.vmc.o'

up=np.zeros((9,9))
uperr=np.zeros((9,9))
dn=np.zeros((9,9))
dnerr=np.zeros((9,9))

with open(fname,'r') as f:
  for line in f:
    if "One-body density matrix" in line:
      for i in range(82):
        l=next(f)
        if(i>0):
          dl=l.split(" ")
          data=[]
          for x in dl:
            if(x!=''):data.append(x)
          data=[float(x) for x in data]
          up[int(data[0]),int(data[1])]=data[2]
          uperr[int(data[0]),int(data[1])]=data[3]
          dn[int(data[0]),int(data[1])]=data[4]
          dnerr[int(data[0]),int(data[1])]=data[5]

labels=["4s","3dxy","3dyz","3dz2","3dxz","3dx2y2","2px","2py","2pz"]
plt.matshow((up+dn)-np.diag(np.diag(up+dn)),vmin=-1,vmax=1,cmap=plt.cm.bwr)
plt.xticks(np.arange(9),labels,rotation=90)
plt.yticks(np.arange(9),labels)
plt.colorbar()
plt.show()
