import os 

def makejson(detgen,N,Ndet,gsw,basename):
  for state in ['','2']:
    for j in range(1,N+1):
      f=basename+'/ex'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc'
      os.system('~/mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  detgen='s'
  N=50
  Ndet=10
  gsw=0.7
  basename='run1s'
  makejson(detgen,N,Ndet,gsw,basename)
