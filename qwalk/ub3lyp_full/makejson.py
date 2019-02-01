import os 
import numpy as np 

def makejson(N,gsw,basename):
  for j in range(1,N+1):
    f=basename+'/gsw'+str(np.around(gsw,2))+'_'+str(j)+'.vmc'
    #if(os.path.isfile(f+'.gosling.json')): print(f+'.gosling.json exists') 
    #else:
    print('Running '+f+'.gosling.json')
    os.system('../../../mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  N=10
  for basestate in np.arange(16):
    #for gsw in np.arange(0.6,1.0,0.1):
    for gsw in [0.3]:  
      if(gsw==1.0): N=1
      else: N=10
      makejson(N,gsw,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
