import sys
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
import numpy as np 
import matplotlib.pyplot as plt 
from analyze_jsonlog import gather_json_df
import json 
from scipy.stats import probplot

'''
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
        #if("Wave function derivatives" in line): 
        if("Derivative of the energy" in line):
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
  plt.ylabel("dpenergy1")

  plt.subplot(212)
  plt.plot(cutoff,frac*100,'-o')
  plt.xlabel("Cutoff")
  plt.ylabel("Percent steps dropped")
#plt.show()
plt.savefig('analyze_all1r_e.pdf')
plt.close()
'''

#analyze_all2.pdf
#cutoff=["0","0.25","0.5","0.6","0.65","0.7","0.75","0.775","0.8"]
#cutoff=['0', '0.0001', '0.00012', '0.00014', '0.00016', '0.00018', '0.0002', '0.00022', '0.00024', '0.00026', '0.00028', '0.0003'] 
cutoff=['0', '0.0001', '0.00012', '0.00014', '0.00016', '0.00018', '0.0002', '0.00022', '0.00024', '0.00026', '0.00028', '0.0003',
'0.00035', '0.00045', '0.0005', '0.00055', '0.00065', '0.0007', '0.00075', '0.0008', '0.00085', '0.0009', '0.00095', '0.001']
count=8
#for myf in ["/u/sciteam/sapatha2//scratch/nchoose10_sgn2/all"+str(i)+"/Cuvtz0_B3LYP_s3_g0.1_cAll.vmc" for i in range(1,7)]:
for myf in ["/u/sciteam/sapatha2//scratch/nchoose10_sgn2/all"+str(i)+"/Cuvtz0_B3LYP_s3_g0.1_cAll.vmc" for i in range(8,9)]:
  count+=1
  blocki=-1
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
            dpwfblocks[i].append(block['derivative_dm'][cutoff[i]]['dpwf']['vals'][0])
            dpeblocks[i].append(block['derivative_dm'][cutoff[i]]['dpenergy']['vals'][0])
            eblocks[i].append(block['total_energy']['value'][0])
            drop[i].append(block['derivative_dm'][cutoff[i]]['drop'])
  dpwfblocks=np.array(dpwfblocks)
  dpeblocks=np.array(dpeblocks)
  eblocks=np.array(eblocks)
  drop=np.array(drop)  
  for j in range(len(cutoff)):
    #print(np.mean(dpwfblocks[j]),np.std(dpwfblocks[j])/np.sqrt(len(dpwfblocks[j])))
    #plt.errorbar(np.mean(drop[j])*100,np.mean(dpwfblocks[j]),yerr=np.std(dpwfblocks[j]/np.sqrt(len(dpwfblocks[j]))),fmt='ob')
    probplot(dpwfblocks[j],plot=plt)
    plt.savefig('analyze_all2_'+cutoff[j]+'_'+str(count)+'.png')
    plt.close()
  #plt.show() 
