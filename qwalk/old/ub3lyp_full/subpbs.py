import os 
import numpy as np 

def subpbs(N,gsw,basestate,basename):
  fout='gsw'+str(np.round(gsw,2))
  for j in range(1,N+1):
    fname=basename+'/'+fout+'_'+str(j)+'.pbs'
    os.system('qsub '+fname)

if __name__=='__main__':
  #for gsw in np.arange(1.0,0.0,-0.1):
  for gsw in [0.1,0.2]:
    if(gsw==1.0): N=1
    else: N=10
    for basestate in np.arange(16):
      subpbs(N,gsw,basestate,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
