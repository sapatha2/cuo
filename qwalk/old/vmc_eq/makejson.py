import os 

def makejson(detgen,N,Ndet,gsw,basename):
  for state in ['gs0','gs1','gs2','gs3','gs4','gs5','gs2_lo','gs3_lo','gs5_lo']:
    for j in range(1,N+1):
      f=basename+'/'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc'
      os.system('../../../mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  detgen='s'
  N=100
  Ndet=10
  gsw=0.7
  basename='run2s'
  makejson(detgen,N,Ndet,gsw,basename)
