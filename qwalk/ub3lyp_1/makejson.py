import os 
import numpy as np 

def makejson(N,gsw,basename):
  for j in range(1,N+1):
    f=basename+'/gsw'+str(np.around(gsw,2))+'_'+str(j)+'.vmc'
    print(f)
    os.system('../../../mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  
    #Replace labels in JSON 
    json_f=open(f+'.gosling.json','r').read().split("\n")
    i=1 
    for j in range(len(json_f)):
      if('tbdm_basis' in json_f[j]): 
        json_f[j]='"tbdm_basis'+str(i)+'":{'
        i+=1
    json_o=open(f+'.gosling.json','w')
    json_o.write('\n'.join(json_f))
  return 1

if __name__=='__main__':
  for basestate in np.arange(11,13):
    #for gsw in np.arange(0.1,1.1,0.1):
    for gsw in [1.0]:
      N=10
      if(gsw==1.0): N=1      
      makejson(N,gsw,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
