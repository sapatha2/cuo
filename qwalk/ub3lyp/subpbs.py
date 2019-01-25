import os 
import numpy as np 

def subpbs(N,gsw,basename):
  fout='gsw'+str(np.round(gsw,2))
  for j in range(1,N+1):
    fname=basename+'/'+fout+'_'+str(j)+'.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  N=40
  for gsw in np.arange(0.1,1.0,0.1):
    subpbs(N,gsw,basename='gsw'+str(np.around(gsw,2)))
