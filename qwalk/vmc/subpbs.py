import os 
import numpy as np 

def subpbs(detgen,N,Ndet,gsw,basename):
  fout='_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(np.round(gsw,2))
  for state in ['','2']:
    for j in range(1,N+1):
      fname=basename+'/ex'+state+fout+'_'+str(j)+'.pbs'
      os.system('qsub '+fname)

if __name__=='__main__':
  detgen='a'
  N=50
  Ndet=10
  gsw=0.9
  subpbs(detgen,N,Ndet,gsw,basename='run1a/')
