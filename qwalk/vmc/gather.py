import json 
import pandas as pd
import numpy as np 
import sys
sys.path.append('../../../downfolding/')
from downfold_tools import get_qwalk_dm, sum_onebody, sum_J, sum_U, sum_V

def gather_all(detgen,N,Ndet,gsw,basename):
  ''' 
  Gathers all your data and stores into 
  '''
  for state in ['','2']:
    #Ground states
    f='base/gs'+state+'.vmc_tbdm.gosling.json'
    obdm,__,tbdm__=get_qwalk_dm(json.load(open(f,'r')))

    #Excited states
    for j in range(1,N+1):
      f=basename+'/ex'+state+'_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(gsw)+'_'+str(j)+'.vmc.gosling.json'
      obdm,__,tbdm__=get_qwalk_dm(json.load(open(f,'r')))
  
  return 1

if __name__=='__main__':
  detgen='a'
  N=50
  Ndet=10
  gsw=0.8
  basename='run2a'
  gather_all(detgen,N,Ndet,gsw,basename)
