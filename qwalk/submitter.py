#Submit particular files
import os 

N=20 #Number of files to submit 

for cutoff in ["0p2","0p3","0p4"]:
  for i in range(1,N+1):
    fname="Cuvtz0_B3LYP.dmc_E_SR_"+str(i)+cutoff+".pbs" 
    os.system("qsub "+fname)


