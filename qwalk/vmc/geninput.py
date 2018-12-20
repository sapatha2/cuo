#Generate input files 
import os 
import shutil 
import numpy as np 

def geninput(detgen,N,Ndet,gsw,basename):
  '''
  input:
  detgen - type of excitations to sample
  N - number of states to sample per spin (2*N states in total)
  Ndet - number of determinants per state
  gsw - ground state weight 
  basename - all written files to basename/__generated file names__
  output:
  slater files
  vmc files 
  pbs files for vmc
  '''
  
  #Make sure directory exists, else make it 
  if not os.path.exists(basename):
    #Move all important files to there 
    shutil.copytree('req_files',basename+'/') 
  else:
    print('Directory '+str(basename)+' exists, not going to overwrite')
    exit(0)  

  fout='_'+detgen+'_Ndet'+str(Ndet)+'_gsw'+str(np.round(gsw,2))
  genslater(N,Ndet,gsw,basename,fout)
  genvmc(N,basename,fout)
  genpbs(N,basename,fout)
  return 

def genpbs(N,basename,fout):
  for state in ['','2']:
    for j in range(1,N+1):
      fname='ex'+state+fout+'_'+str(j)
      
      string='#!/bin/bash\n'+\
      '#PBS -q low\n'+\
      '#PBS -l nodes=2:ppn=32:xe\n'+\
      '#PBS -l walltime=02:00:00\n'+\
      '#PBS -N '+fname+'\n'\
      '#PBS -e '+fname+'.perr\n'+\
      '#PBS -o '+fname+'.pout\n'+\
      'mkdir -p /scratch/sciteam/$USER/cuo/qwalk/vmc/'+str(basename)+'\n'+\
      'cd /scratch/sciteam/$USER/cuo/qwalk/vmc/'+str(basename)+'\n'+\
      'cp -u /u/sciteam/$USER/cuo/qwalk/vmc/'+str(basename)+'/* .\n'+\
      'aprun -n 64 /u/sciteam/$USER/fork/bin/qwalk '+fname+'.vmc &> '+fname+'.vmc.out\n'
      
      f=open(basename+'/'+fname+'.pbs','w')
      f.write(string)
      f.close()      
  return 1

def genvmc(N,basename,fout):
  for state in ['','2']:
    for j in range(1,N+1):
      fname='ex'+state+fout+'_'+str(j)
      
      string='method {\n'+\
      '  vmc\n'+\
      '  nblock 2000\n'+\
      '  average { tbdm_basis\n'+\
      '    mode obdm\n'+\
      '    orbitals {\n'+\
      '      magnify 1\n'+\
      '      nmo 14\n'+\
      '      orbfile iao.orb\n'+\
      '      include iao.basis\n'+\
      '      centers { useglobal }\n'+\
      '    }\n'+\
      '    states { 2 6 7 8 9 10 12 13 14 }\n'+\
      '  }\n'+\
      '}\n'+\
      '\n'+\
      'include gs'+state+'.sys\n'+\
      'trialfunc {\n'+\
      '  slater-jastrow\n'+\
      '  wf1 { include '+fname+'.slater }\n'+\
      '  wf2 { include gs.optjast3 }\n'+\
      '}\n'

      f=open(basename+'/'+fname+'.vmc','w')
      f.write(string)
      f.close()      
  return 1

def genslater(N,Ndet,gsw,basename,fout):
  return -1 

if __name__=='__main__':
  geninput('s',100,10,0.7,basename='run1/')
