import os 
import numpy as np 

def makejson(N,gsw,basename):
  for j in range(1,N+1):
    f=basename+'/gsw'+str(np.around(gsw,2))+'_'+str(j)+'.vmc'
    print(f)
    os.system('../../../mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  N=10
  for basestate in np.arange(4,5):
    for gsw in np.arange(0.1,1.0,0.1):
      makejson(N,gsw,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
