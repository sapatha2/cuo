import numpy as np 
import matplotlib.pyplot as plt 

cutoff=[]
dpwf1=[]
dpwf1err=[]
frac=[]
read=0
with open("Cuvtz0_B3LYP_s3_g0.1_cAll.vmc.o") as f:
  for line in f:
    if("Run Ended" in line): read=1
    if(read): 
      if("Cutoff" in line): cutoff.append(float(line.split(" ")[1]))
      if("Wave function derivatives" in line): 
        for i in range(2): d=next(f)
        dpwf1.append(float(d.split("+/-")[0]))
        dpwf1err.append(float(d.split("+/-")[1]))
      if("Fraction" in line): frac.append(float(line.split(":")[1]))

'''
print(cutoff)
print(frac)
print(dpwf1)
print(dpwf1err)
'''

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
plt.show()
