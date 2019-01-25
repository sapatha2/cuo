import os 

def makejson(detgen,N,Ndet,gsw,basename):
<<<<<<< HEAD
  for state in ['2Y','4SigmaM']:
=======
  for state in ['2X']:
>>>>>>> 25bfdb4a0a27ebfa182c5df1472ce635d5a718b6
    for j in range(1,N+1):
      f=basename+'/'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc'
      os.system('../../..//mainline/bin/gosling '+f+'.log -json &> '+f+'.gosling.json')
  return 1

if __name__=='__main__':
  detgen='a'
  N=200
  Ndet=10
  gsw=0.9
  basename='run1a'
  makejson(detgen,N,Ndet,gsw,basename)
