#Generate input files for 2rdm, use same slater files as old 1rdm!
import os 
import shutil 
import numpy as np 

def geninput(N,gsw,basestate,basename):
  '''
  input:
  detgen - type of excitations to sample
  N - number of states to sample
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
    print('Directory '+str(basename)+' doesnt exists, not going to write DMC')
    exit(0)  
  fout='gsw'+str(np.round(gsw,2))
  gendmc(N,basename,fout)
  genpbs(N,basename,fout)
  return 

def genpbs(N,basename,fout):
  for j in range(1,N+1):
    fname=fout+'_'+str(j)
   
    #Blue waters input  
    string='#!/bin/bash\n'+\
    '#PBS -q secondary\n'+\
    '#PBS -l nodes=1,flags=allprocs\n'+\
    '#PBS -l walltime=03:00:00\n'+\
    '#PBS -N '+fname+str(j)+'\n'\
    '#PBS -e '+fname+str(j)+'.perr\n'+\
    '#PBS -o '+fname+str(j)+'.pout\n'+\
    'module load openmpi/3.1.1-gcc-7.2.0\n'+\
    'module load intel/18.0\n'+\
    'cd ${PBS_O_WORKDIR}/'+basename+'/ \n'+\
    'mpiexec ../../../../mainline/bin/qwalk '+fname+'.dmc &> '+fname+'.dmc.out\n'
    
    f=open(basename+'/'+fname+'.dmc.pbs','w')
    f.write(string)
    f.close()      
  return 1

def gendmc(N,basename,fout):
  for j in range(1,N+1):
    sysstring='s12.sys'
    fname=fout+'_'+str(j)
    
    string='method {\n'+\
    '  dmc\n'+\
    '  nblock 25\n'+\
    '  timestep 0.01 \n'+\
    '  tmoves \n'+\
    '  average { tbdm_basis\n'+\
    '    mode obdm\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 14\n'+\
    '      orbfile gs1.orb\n'+\
    '      include gs1.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 5 6 7 8 9 10 11 12 13 14 }\n'+\
    '  }\n'+\
    '}\n'+\
    '\n'+\
    'include '+sysstring+'\n'+\
    'trialfunc {\n'+\
    '  slater-jastrow\n'+\
    '  wf1 { include '+fname+'.slater }\n'+\
    '  wf2 { include optjast3 }\n'+\
    '}\n'

    f=open(basename+'/'+fname+'.dmc','w')
    f.write(string)
    f.close()      
  return 1

if __name__=='__main__':
  for gsw in np.arange(0.1,1.1,0.1):
    if(gsw==1.0): N=1
    else: N=10
    for basestate in np.arange(10):
      geninput(N,gsw,basestate,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
