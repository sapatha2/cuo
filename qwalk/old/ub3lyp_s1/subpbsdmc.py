import os 
import numpy as np 

def subpbs(N,gsw,basestate,basename):
  fout='gsw'+str(np.round(gsw,2))
  for j in range(1,N+1):
    fname=basename+'/'+fout+'_'+str(j)+'.dmc.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  for basestate in np.arange(10):
    for gsw in np.arange(0.2,1.2,0.2):
      if(gsw==1.0): N=1
      else: N=5
      subpbs(N,gsw,basestate,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
