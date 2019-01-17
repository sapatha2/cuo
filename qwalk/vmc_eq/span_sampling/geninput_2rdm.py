#Generate input files for 2rdm, use same slater files as old 1rdm!
import os 
import shutil 
import numpy as np 

def geninput(spin,N,gsw,basename):
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
  fout='S'+str(spin)+'_gsw'+str(np.round(gsw,2))
  genslater(spin,N,gsw,basename,fout)
  genvmc(spin,N,basename,fout)
  genpbs(N,basename,fout)
  return 

def genpbs(N,basename,fout):
  for j in range(1,N+1):
    fname=fout+'_'+str(j)
   
    #Blue waters input  
    string='#!/bin/bash\n'+\
    '#PBS -q secondary\n'+\
    '#PBS -l nodes=1,flags=allprocs\n'+\
    '#PBS -l walltime=00:30:00\n'+\
    '#PBS -N '+fname+'\n'\
    '#PBS -e '+fname+'.perr\n'+\
    '#PBS -o '+fname+'.pout\n'+\
    'module load openmpi/3.1.1-gcc-7.2.0\n'+\
    'module load intel/18.0\n'+\
    'cd ${PBS_O_WORKDIR}/'+basename+'/ \n'+\
    'mpiexec ../../../../../mainline/bin/qwalk '+fname+'.vmc &> '+fname+'.vmc.out\n'
    
    f=open(basename+'/'+fname+'.pbs','w')
    f.write(string)
    f.close()      
  return 1

def genvmc(spin,N,basename,fout):
  for j in range(1,N+1):
    sysstring='s12.sys'
    if(spin==3): sysstring='s32.sys'
    fname=fout+'_'+str(j)
    
    string='method {\n'+\
    '  vmc\n'+\
    '  nblock 100\n'+\
    '  average { tbdm_basis\n'+\
    '    orbitals {\n'+\
    '      magnify 1\n'+\
    '      nmo 14\n'+\
    '      orbfile b3lyp_iao_b.orb\n'+\
    '      include b3lyp_iao_b.basis\n'+\
    '      centers { useglobal }\n'+\
    '    }\n'+\
    '    states { 2 6 7 8 9 10 12 13 14 }\n'+\
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

def genslater(spin,N,gsw,basename,fout):
  for j in range(1,N+1):
    fname=fout+'_'+str(j)
    #Generate input file, based on following state order
    #(see collect_mos.py for ordering): gs0, gs1, gs2, gs3, gs4, gs5
    string=None
    if(spin==1):
      Ndet=6
      
      #Generate weight vector 
      gauss=np.random.normal(size=Ndet-1)
      gauss/=np.sqrt(np.dot(gauss,gauss))
      w=np.zeros(Ndet)+np.sqrt(gsw)
      w[1:]=gauss*np.sqrt(1-gsw)
      
      #Make input file
      string='SLATER\n'+\
      'ORBITALS  {\n'+\
      '  MAGNIFY 1.0\n'+\
      '  NMO 88\n'+\
      '  ORBFILE all.orb\n'+\
      '  INCLUDE all.basis\n'+\
      '  CENTERS { USEGLOBAL }\n'+\
      '}\n'+\
      '\n'+\
      'DETWT { \n' + '\n'.join(['  '+str(x) for x in w])+' \n}\n'+\
      'STATES {\n'+\
      '  1 2 3 4 5 6 7 8 9 10 11 12 13\n'+\
      '  1 2 3 4 5 6 7 8 9 10 11 12   \n\n'+\
      '  15 16 17 18 19 20 21 22 23 24 25 26 27\n'+\
      '  15 16 17 18 19 20 21 22 23 24 25 26   \n\n'+\
      '  29 30 31 32 33 34 35 36 37 38 39 40 41\n'+\
      '  29 30 31 32 33 34 35 36 37 38 39 42   \n\n'+\
      '  43 44 45 46 47 48 49 50 51 52 53 54 55\n'+\
      '  43 44 45 46 47 48 49 50 51 52 53 56   \n\n'+\
      '  57 58 59 60 61 62 63 64 65 66 67 68 69\n'+\
      '  57 58 59 60 61 62 63 64 65 66 67 68   \n\n'+\
      '  71 72 73 74 75 76 77 78 79 80 81 82 83\n'+\
      '  71 72 73 74 75 76 77 78 79 80 81 84   \n\n'+\
      '}\n'
    elif(spin==3):
      Ndet=3
      
      #Generate weight vector 
      gauss=np.random.normal(size=Ndet-1)
      gauss/=np.sqrt(np.dot(gauss,gauss))
      w=np.zeros(Ndet)+np.sqrt(gsw)
      w[1:]=gauss*np.sqrt(1-gsw)
      
      #Make input file
      string='SLATER\n'+\
      'ORBITALS  {\n'+\
      '  MAGNIFY 1.0\n'+\
      '  NMO 88\n'+\
      '  ORBFILE all.orb\n'+\
      '  INCLUDE all.basis\n'+\
      '  CENTERS { USEGLOBAL }\n'+\
      '}\n'+\
      '\n'+\
      'DETWT { \n' + '\n'.join(['  '+str(x) for x in w])+' \n}\n'+\
      'STATES {\n'+\
      '  29 30 31 32 33 34 35 36 37 38 39 40 41 42\n'+\
      '  29 30 31 32 33 34 35 36 37 38 39         \n\n'+\
      '  43 44 45 46 47 48 49 50 51 52 53 54 55 56\n'+\
      '  43 44 45 46 47 48 49 50 51 52 53         \n\n'+\
      '  71 72 73 74 75 76 77 78 79 80 81 82 83 84\n'+\
      '  71 72 73 74 75 76 77 78 79 80 81         \n\n'+\
      '}\n'
    else:
      print("Can't do that spin")
      exit(0)

    f=open(basename+'/'+fname+'.slater','w')
    f.write(string)
    f.close()
  return 1

if __name__=='__main__':
  N=20
  spin=1 #2*Sz
  for gsw in np.arange(0.1,1.0,0.1):
    geninput(spin,N,gsw,basename='spin'+str(spin)+'gsw'+str(np.around(gsw,2)))
