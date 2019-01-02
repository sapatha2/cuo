import os 
import numpy as np 

def subpbs(detgen,N,Ndet,gsw,basename):
  fout='_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(np.round(gsw,2))
  for state in ['2X','2Y','4SigmaM','4Phi','4Delta','2Delta','4SigmaP']:
    for j in range(1,N+1):
      fname=basename+'/'+state+fout+'_'+str(j)+'.pbs'
      os.system('qsub '+fname)

if __name__=='__main__':
  detgen='a'
  N=25
  Ndet=10
  gsw=0.8
  subpbs(detgen,N,Ndet,gsw,basename='run1a/')
