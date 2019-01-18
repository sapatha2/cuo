import os 
import numpy as np 

def makejson(spin,N,gsw,basename):
  for j in range(1,N+1):
    f=basename+'/S'+str(spin)+'_gsw'+str(np.around(gsw,2))+'_'+str(j)+'.vmc'
    print(f)
    os.system('../../../../mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  N=20
  spin=3
  for gsw in np.arange(0.1,1.0,0.1):
    makejson(spin,N,gsw,basename='spin'+str(spin)+'gsw'+str(np.around(gsw,2)))
