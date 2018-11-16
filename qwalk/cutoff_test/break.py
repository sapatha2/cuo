import sys
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
import numpy as np 
import matplotlib.pyplot as plt 
from analyze_jsonlog import gather_json_df
import json 
from scipy.stats import probplot

#Figures
cutoff=['0', '0.0001', '0.00012', '0.00014', '0.00016', '0.00018', '0.0002', '0.00022', '0.00024', '0.00026', '0.00028', '0.0003',
'0.00035', '0.00045', '0.0005', '0.00055', '0.00065', '0.0007', '0.00075', '0.0008', '0.00085', '0.0009', '0.00095', '0.001']

p=12
parm=9
myf="/u/sciteam/sapatha2/scratch/nchoose10_sgn2/all"+str(p)+"/Cuvtz0_B3LYP_s3_g0.1_cAll.vmc"
dpwfblocks=[[] for i in range(len(cutoff))]
dpeblocks= [[] for i in range(len(cutoff))] 
eblocks=   [[] for i in range(len(cutoff))]
drop=      [[] for i in range(len(cutoff))]
with open(myf+".json") as jsonf:
  for blockstr in jsonf.read().split("<RS>"):
    if '{' in blockstr:
      to_read=blockstr.replace("inf","0")
      to_read=to_read.replace("},\n}","}\n}")
      block = json.loads(to_read)['properties']
      keys=list(block['derivative_dm'].keys())
      if(len(set(keys).intersection(set(cutoff)))>1): 
        for i in range(len(cutoff)):
          dpwfblocks[i].append(block['derivative_dm'][cutoff[i]]['dpwf']['vals'][parm])
          dpeblocks[i].append(block['derivative_dm'][cutoff[i]]['dpenergy']['vals'][parm])
          eblocks[i].append(block['total_energy']['value'][0])
          drop[i].append(block['derivative_dm'][cutoff[i]]['drop'])
dpwfblocks=np.array(dpwfblocks)
dpeblocks=np.array(dpeblocks)
eblocks=np.array(eblocks)
drop=np.array(drop)  

for n in [1000,500,200,100,1]:
  print(n)
  dpwf=[[] for i in range(len(cutoff))]
  dpe= [[] for i in range(len(cutoff))] 
  e=   [[] for i in range(len(cutoff))]
  d=   [[] for i in range(len(cutoff))]
  for j in range(len(cutoff)):
    dpwf[j].append(list(np.split(dpwfblocks[j],n)))
    dpe[j].append(list(np.split(dpeblocks[j],n)))
    e[j].append(list(np.split(eblocks[j],n)))
    d[j].append(list(np.split(drop[j],n)))
  dpwf=np.array(dpwf)
  dpe=np.array(dpe)
  e=np.array(e)
  d=np.array(d)

  for i in range(n):
    #probplot(dpe[0,0,i,:],plot=plt) 
    #plt.show()
    #x=np.mean(d[:,0,i,:],axis=1)
    if(min(dpe[0,0,i,:])<-100):
      x=cutoff
      y=np.mean(dpe[:,0,i,:],axis=1)
      yerr=np.std(dpe[:,0,i,:],axis=1)/np.sqrt(dpe.shape[3])
      plt.errorbar(x,y,yerr=yerr,fmt='o-',label=str(20000/n))
plt.xticks(rotation=90)
plt.legend(loc=1)
plt.show()

'''
b=2 #Block to use 
one=dpe[:,0,b,:]
two=dpwf[:,0,b,:]
three=e[:,0,b,:]
dHdp=[]
dHdperr=[]

#Cutoff loop
for i in range(one.shape[0]):
  #Bootstrap block
  tmp=[]
  for p in range(100): 
    o=np.mean(np.random.choice(one[i,:],dpe.shape[3]))
    t=np.mean(np.random.choice(two[i,:],dpe.shape[3]))
    th=np.mean(np.random.choice(three[i,:],dpe.shape[3]))
    tmp.append(o-t*th) 
  dHdp.append(np.mean(tmp))
  dHdperr.append(np.std(tmp))

plt.errorbar(cutoff,dHdp,yerr=dHdperr,fmt='o-')
plt.show()
'''
