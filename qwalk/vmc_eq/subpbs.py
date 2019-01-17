import os 
import numpy as np 

def subpbs(detgen,N,Ndet,gsw,basename):
  fout='_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(np.round(gsw,2))
  #for state in ['gs0','gs1','gs2','gs3','gs4','gs5','gs2_lo','gs3_lo','gs5_lo']:
  for state in ['gs2_lo','gs3_lo','gs5_lo']:
    for j in range(1,N+1):
      fname=basename+'/'+state+fout+'_'+str(j)+'.pbs'
      os.system('qsub '+fname)

if __name__=='__main__':
  detgen='a'
  N=100
  Ndet=10
  gsw=0.7
  subpbs(detgen,N,Ndet,gsw,basename='run2a/')
