import os 
import numpy as np 

def subpbs(N,gsw,basestate,basename):
  fout='gsw'+str(np.round(gsw,2))
  for j in range(1,N+1):
    fname=basename+'/'+fout+'_'+str(j)+'.dmc.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  for basestate in np.arange(10):
    for gsw in [1.0]:
      subpbs(1,gsw,basestate,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
