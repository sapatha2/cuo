import os 

def makejson(detgen,N,Ndet,gsw,basename):
  for state in ['2X','2Y','4SigmaM','4Delta','4Phi','2Delta','4SigmaP']:
    for j in range(1,N+1):
      f=basename+'/'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc'
      os.system('~/mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  detgen='a'
  N=25
  Ndet=10
  gsw=0.8
  basename='run1a'
  makejson(detgen,N,Ndet,gsw,basename)
