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
    #BW input
    i=1
    fname=fout+'_'+str(i)
    string='#!/bin/bash\n'+\
    '#PBS -q low\n'+\
    '#PBS -l nodes=1:ppn=32:xe\n'+\
    '#PBS -l walltime=04:00:00\n'+\
    '#PBS -N '+fname+'\n'\
    '#PBS -e '+fname+'.perr\n'+\
    '#PBS -o '+fname+'.pout\n'+\
    'mkdir -p /scratch/sciteam/$USER/cuo/qwalk/ub3lyp_base_s1/'+basename+'/\n'+\
    'cd /scratch/sciteam/$USER/cuo/qwalk/ub3lyp_base_s1/'+basename+'/\n'+\
    'cp -u /u/sciteam/$USER/cuo/qwalk/ub3lyp_base_s1/'+basename+'/* .\n'+\
    'aprun -n 32 /u/sciteam/$USER/fork/bin/qwalk '+fname+'.vmc &> '+fname+'.vmc.out\n'   

    f=open(basename+'/'+fname+'.pbs','w')
    f.write(string)
    f.close()
  else: pass
  return 1

def genvmc(N,basename,fout):
  for j in range(1,N+1):
    sysstring='s12.sys'
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

def genslater(N,gsw,basestate,basename,fout):
  for j in range(1,N+1):
    fname=fout+'_'+str(j)
    #Generate input file, based on following state order
    #(see collect_mos.py for ordering): gs0, gs1, gs2, gs3, gs4, gs5
    string=None
    Ndet=11

    #Generate weight vector 
    gauss=np.random.normal(size=Ndet-1)
    gauss/=np.sqrt(np.dot(gauss,gauss))
    w=np.zeros(Ndet)
    w[basestate]=np.sqrt(gsw) 
    w[w==0]=gauss*np.sqrt(1-gsw)
    assert(abs(np.dot(w,w)-1)<1e-15)

    states_up=np.arange(1,14)
    states_dn=np.arange(1,13)+Ndet*14
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
      states_up=np.arange(1,14)
      states_dn=np.arange(1,13)+Ndet*14
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
    '  ORBFILE all_1do.orb\n'+\
    '  INCLUDE all_1do.basis\n'+\
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
  for gsw in [1.0]:
    for basestate in np.arange(10):
      geninput(1,gsw,basestate,basename='gsw'+str(np.around(gsw,2))+'b'+str(basestate))
