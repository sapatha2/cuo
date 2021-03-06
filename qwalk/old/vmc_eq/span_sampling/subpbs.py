import os 
import numpy as np 

def subpbs(spin,N,gsw,basename):
  fout='S'+str(spin)+'_gsw'+str(np.round(gsw,2))
  for j in range(1,N+1):
    fname=basename+'/'+fout+'_'+str(j)+'.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  N=20
  spin=3 #2*Sz
  for gsw in np.arange(0.1,1.0,0.1):
    subpbs(spin,N,gsw,basename='spin'+str(spin)+'gsw'+str(np.around(gsw,2)))
