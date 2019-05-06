#Generate input files for 2rdm, use same slater files as old 1rdm!
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
  genslater(detgen,N,Ndet,gsw,basename,fout)
  genvmc(N,basename,fout)
  genpbs(N,basename,fout)
  return 

def genpbs(N,basename,fout):
  for state in ['2X','2Y','4SigmaM']:
    for j in range(1,N+1):
      fname=state+fout+'_'+str(j)
     
      #Blue waters input  
      string='#!/bin/bash\n'+\
      '#PBS -q wagner\n'+\
      '#PBS -l nodes=1,flags=allprocs\n'+\
      '#PBS -l walltime=00:10:00\n'+\
      '#PBS -N '+fname+'\n'\
      '#PBS -e '+fname+'.perr\n'+\
      '#PBS -o '+fname+'.pout\n'+\
      'module load openmpi/3.1.1-gcc-7.2.0\n'+\
      'module load intel/18.0\n'+\
      'cd ${PBS_O_WORKDIR}/'+basename+'\n'+\
      'mpiexec ../../../../mainline/bin/qwalk '+fname+'.vmc &> '+fname+'.vmc.out\n'

      f=open(basename+'/'+fname+'.pbs','w')
      f.write(string)
      f.close()      
  return 1

def genvmc(N,basename,fout):
  for state in ['2X','2Y','4SigmaM']:
    for j in range(1,N+1):
      fname=state+fout+'_'+str(j)
      
      string='method {\n'+\
      '  vmc\n'+\
      '  nblock 100\n'+\
      '  average { tbdm_basis\n'+\
      '    mode obdm\n'+\
      '    orbitals {\n'+\
      '      magnify 1\n'+\
      '      nmo 14\n'+\
      '      orbfile mo_opt.orb\n'+\
      '      include mo_opt.basis\n'+\
      '      centers { useglobal }\n'+\
      '    }\n'+\
      '    states { 6 7 8 11 12 13 14 }\n'+\
      '  }\n'+\
      '  average { tbdm_basis\n'+\
      '    orbitals {\n'+\
      '      magnify 1\n'+\
      '      nmo 14\n'+\
      '      orbfile iao.orb\n'+\
      '      include iao.basis\n'+\
      '      centers { useglobal }\n'+\
      '    }\n'+\
      '    states { 2 7 8 9 } \n'+\
      '  }\n'+\
      '}\n'+\
      'include '+state+'.sys\n'+\
      'trialfunc {\n'+\
      '  slater-jastrow\n'+\
      '  wf1 { include '+fname+'.slater }\n'+\
      '  wf2 { include 2X.optjast3 }\n'+\
      '}\n'

      f=open(basename+'/'+fname+'.vmc','w')
      f.write(string)
      f.close()      
  return 1

def genslater(detgen,N,Ndet,gsw,basename,fout):
  occ={'2X':[13,12],'2Y':[13,12],'4SigmaM':[14,11]}
  act=[6,7,8,11,12,13,14] #Active space is independent of spin state
  core=[1,2,3,4,5,9,10]
  for state in ['2X','2Y','4SigmaM']:
    for j in range(1,N+1):
      fname=state+fout+'_'+str(j)

      #Generate weight vector 
      gauss=np.random.normal(size=Ndet-1)
      gauss/=np.sqrt(np.dot(gauss,gauss))
      w=np.zeros(Ndet)+np.sqrt(gsw)
      w[1:]=gauss*np.sqrt(1-gsw)

      #Generate determinants
      detstring=''
      detstring+='  '+' '.join([str(x) for x in range(1,1+occ[state][0])])+'\n'
      detstring+='  '+' '.join([str(x) for x in range(1,1+occ[state][1])])+'\n\n'

      for j in range(1,Ndet):
        #Full active space
        if(detgen=='a'):
          r=np.random.choice(act,size=occ[state][0]-len(core),replace=False)
          s=np.random.choice(act,size=occ[state][1]-len(core),replace=False)
          r=list(r)
          s=list(s)
          r+=core
          s+=core
          r=sorted(r)
          s=sorted(s)
          r=[str(x) for x in list(r)]
          s=[str(x) for x in list(s)]
          detstring+='  '+' '.join(r)+'\n'+'  '+' '.join(s)+'\n\n'
        else:
          print("Detgen = "+str(a)+" isn't ready.")
          exit(0)

      #Generate input file 
      string='SLATER\n'+\
      'ORBITALS  {\n'+\
      '  MAGNIFY 1.0\n'+\
      '  NMO 14\n'+\
      '  ORBFILE '+state+'.orb\n'+\
      '  INCLUDE '+state+'.basis\n'+\
      '  CENTERS { USEGLOBAL }\n'+\
      '}\n'+\
      '\n'+\
      'DETWT { \n' + '\n'.join(['  '+str(x) for x in w])+' \n}\n'+\
      'STATES {\n'+\
      detstring[:-1]+\
      '}\n'

      f=open(basename+'/'+fname+'.slater','w')
      f.write(string)
      f.close()
  return 1

if __name__=='__main__':
  detgen='a'
  N=200
  Ndet=10
  gsw=0.9
  geninput(detgen,N,Ndet,gsw,basename='run2a')
