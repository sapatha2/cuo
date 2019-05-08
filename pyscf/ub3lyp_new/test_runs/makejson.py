import os 
import numpy as np 

def makejson():
  for f in ['Cuvtz_r1.725_s1_UB3LYP_11.chk.vmc',
  'Cuvtz_r1.725_s1_UB3LYP_12.chk','Cuvtz_r1.725_s3_UB3LYP_13.chk']:
    os.system('../../../../mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  
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
  makejson()
