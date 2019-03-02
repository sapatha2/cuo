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
    #Move all important files to there 
    shutil.copytree('req_files',basename+'/') 
  else:
    print('Directory '+str(basename)+' exists, not going to overwrite')
    exit(0)  
  fout='gsw'+str(np.round(gsw,2))
  genslater(N,gsw,basestate,basename,fout)
  genvmc(N,basename,fout)
  genpbs(N,basename,fout)
  return 

def genpbs(N,basename,fout):
  if(N==1):
    #CC input
    i=1
    fname=fout+'_'+str(i)
    string='#!/bin/bash\n'+\
    '#PBS -q secondary\n'+\
    '#PBS -l nodes=1,flags=allprocs\n'+\
    '#PBS -l walltime=01:00:00\n'+\
    '#PBS -N '+fname+'\n'\
    '#PBS -e '+fname+'.perr\n'+\
    '#PBS -o '+fname+'.pout\n'+\
    'module load openmpi/3.1.1-gcc-7.2.0\n'+\
    'module load intel/18.0\n'+\
    'cd ${PBS_O_WORKDIR}/'+basename+'/ \n'+\
    'mpiexec ../../../../mainline/bin/qwalk '+fname+'.vmc &> '+fname+'.vmc.out\n'   
    f=open(basename+'/'+fname+'.pbs','w')
    f.write(string)
    f.close()
  elif(N!=10): print("Cannot run"); exit(0)
  for i in range(2):
    #CC input
    fname=fout+'_'+str(i+1)
    string='#!/bin/bash\n'+\
    '#PBS -q secondary\n'+\
    '#PBS -l nodes=1,flags=allprocs\n'+\
    '#PBS -l walltime=01:00:00\n'+\
    '#PBS -N '+fname+'\n'\
    '#PBS -e '+fname+'.perr\n'+\
    '#PBS -o '+fname+'.pout\n'+\
    'module load openmpi/3.1.1-gcc-7.2.0\n'+\
    'module load intel/18.0\n'+\
    'cd ${PBS_O_WORKDIR}/'+basename+'/ \n'
    for j in range(1+i*5,1+(i+1)*5): #N=10 only!
      frun=fout+'_'+str(j)
      string+='mpiexec ../../../../mainline/bin/qwalk '+frun+'.vmc &> '+frun+'.vmc.out\n'   
    f=open(basename+'/'+fname+'.pbs','w')
    f.write(string)
    f.close()      
  return 1

def genvmc(N,basename,fout):
  for j in range(1,N+1):
    sysstring='s32.sys'
    fname=fout+'_'+str(j)
    
    string='method {\n'+\
    '  vmc\n'+\
    '  nblock 100\n'+\
    '  average { tbdm_basis\n'+\
    '    mode obdm\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 14\n'+\
    '      orbfile gs1.orb\n'+\
    '      include gs1.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 1 2 3 4 5 6 7 8 9 10 11 12 13 14 }\n'+\
    '  }\n'+\
    '  average { tbdm_basis\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 14\n'+\
    '      orbfile iao.orb\n'+\
    '      include iao.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 2 }\n'+\
    '  }\n'+\
    '}\n'+\
    '\n'+\
    'include '+sysstring+'\n'+\
    'trialfunc {\n'+\
    '  slater-jastrow\n'+\
    '  wf1 { include '+fname+'.slater }\n'+\
    '  wf2 { include optjast3 }\n'+\
    '}\n'

    f=open(basename+'/'+fname+'.vmc','w')
    f.write(string)
    f.close()      
  return 1

def genslater(N,gsw,basestate,basename,fout):
  for j in range(1,N+1):
    fname=fout+'_'+str(j)
    #Generate input file, based on following state order
    #(see collect_mos.py for ordering): gs0, gs1, gs2, gs3, gs4, gs5
    string=None
    Ndet=10

    #Generate weight vector 
    gauss=np.random.normal(size=Ndet-1)
    gauss/=np.sqrt(np.dot(gauss,gauss))
    w=np.zeros(Ndet)
    w[basestate]=np.sqrt(gsw) 
    w[w==0]=gauss*np.sqrt(1-gsw)
    assert(abs(np.dot(w,w)-1)<1e-15)

    states_up=np.arange(1,15)
    states_dn=np.arange(1,12)+Ndet*14
    states=[]
    for i in range(Ndet):
      tmp='  '+' '.join([str(x) for x in states_up])+'\n'
      tmp+='  '+' '.join([str(x) for x in states_dn])+'\n\n'
      states.append(tmp)
      states_up+=14
      states_dn+=14
    
    #Specifically for targeted states only
    if(gsw==1.0): 
      w=[1]
      states_up=np.arange(1,15)
      states_dn=np.arange(1,12)+Ndet*14
      states_up+=14*basestate
      states_dn+=14*basestate
      tmp='  '+' '.join([str(x) for x in states_up])+'\n'
      tmp+='  '+' '.join([str(x) for x in states_dn])+'\n\n'
      states=[]
      states.append(tmp)

    #Make input file
    string='SLATER\n'+\
    'ORBITALS  {\n'+\
    '  MAGNIFY 1.0\n'+\
    '  NMO '+str(Ndet*14*2)+'\n'+\
    '  ORBFILE all_hispin.orb\n'+\
    '  INCLUDE all_hispin.basis\n'+\
    '  CENTERS { USEGLOBAL }\n'+\
    '}\n'+\
    '\n'+\
    'DETWT { \n' + '\n'.join(['  '+str(x) for x in w])+' \n}\n'+\
    'STATES {\n'+\
    ''.join(states)+\
    '}\n'
    f=open(basename+'/'+fname+'.slater','w')
    f.write(string)
    f.close()
  return 1

if __name__=='__main__':
  for gsw in np.arange(0.1,1.1,0.1):
    if(gsw==1.0): N=1
    else: N=10
    for basestate in np.arange(6):
      geninput(N,gsw,basestate,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
