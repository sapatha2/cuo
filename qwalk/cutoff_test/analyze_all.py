import sys
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
import numpy as np 
import matplotlib.pyplot as plt 
from analyze_jsonlog import gather_json_df
import json 
from scipy.stats import probplot

#analyze_all1.pdf
for myf in ["/u/sciteam/sapatha2//scratch/nchoose10_sgn2/all"+str(i)+"/Cuvtz0_B3LYP_s3_g0.1_cAll.vmc" for i in [1,2,6]]:
  cutoff=[]
  dpwf1=[]
  dpwf1err=[]
  frac=[]
  read=0
  with open(myf+".o") as f:
    for line in f:
      if("Run Ended" in line): read=1
      if(read): 
        if("Cutoff" in line): cutoff.append(float(line.split(" ")[1]))
        if("Wave function derivatives" in line): 
          for i in range(2): d=next(f)
          dpwf1.append(float(d.split("+/-")[0]))
          dpwf1err.append(float(d.split("+/-")[1]))
        if("Fraction" in line): frac.append(float(line.split(":")[1]))
  cutoff=np.array(cutoff)
  dpwf1=np.array(dpwf1)
  dpwf1err=np.array(dpwf1err)

  frac=np.array(frac)
  plt.subplot(211)
  plt.errorbar(frac*100,dpwf1,yerr=dpwf1err,fmt='o')
  plt.xlabel("Percent steps dropped")
  plt.ylabel("dpwf1")

  plt.subplot(212)
  plt.plot(cutoff,frac*100,'-o')
  plt.xlabel("Cutoff")
  plt.ylabel("Percent steps dropped")
#plt.show()
plt.savefig('analyze_all1r.pdf')

'''
#analyze_all2.pdf
cutoff=["0","0.25","0.5","0.6","0.65","0.7","0.75","0.775","0.8"]
count=0
for myf in ["/u/sciteam/sapatha2//scratch/nchoose10_sgn2/all"+str(i)+"/Cuvtz0_B3LYP_s3_g0.1_cAll.vmc" for i in range(1,7)]:
  count+=1
  dpwfblocks=[[],[],[],[],[],[],[],[],[]]
  with open(myf+".json") as jsonf:
    for blockstr in jsonf.read().split("<RS>"):
      if '{' in blockstr:
        to_read=blockstr.replace("inf","0")
        to_read=to_read.replace("},\n}","}\n}")
        block = json.loads(to_read)['properties']
        for i in range(len(cutoff)):
          dpwfblocks[i].append(block['derivative_dm'][cutoff[i]]['dpwf']['vals'][0])
  dpwfblocks=np.array(dpwfblocks)

  for j in range(len(cutoff)):
    plt.title("Cutoff "+cutoff[j])
    probplot(dpwfblocks[j],plot=plt)
    #plt.show() 
    plt.savefig('analyze_all2_'+str(cutoff[j])+'_'+str(count)+'.png')
    plt.close()
'''
