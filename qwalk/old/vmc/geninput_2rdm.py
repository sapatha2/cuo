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
  for state in ['2X','2Y','4SigmaM','4Phi','4Delta','2Delta','4SigmaP']:
    for j in range(1,N+1):
      fname=state+fout+'_'+str(j)
     
      #Blue waters input  
      string='#!/bin/bash\n'+\
      '#PBS -q low\n'+\
      '#PBS -l nodes=2:ppn=32:xe\n'+\
      '#PBS -l walltime=04:00:00\n'+\
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
  for state in ['2X','2Y','4SigmaM','4Phi','4Delta','2Delta','4SigmaP']:
    for j in range(1,N+1):
      fname=state+fout+'_'+str(j)
      
      string='method {\n'+\
      '  vmc\n'+\
      '  nblock 100\n'+\
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
  occ={'2X':[13,12],'2Y':[13,12],'4SigmaM':[14,11],'4Phi':[14,11],'4Delta':[14,11],'2Delta':[13,12],'4SigmaP':[14,11]}
  act=[6,14] #Active space is independent of spin state
  for state in ['2X','2Y','4SigmaM','4Phi','4Delta','2Delta','4SigmaP']:
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
        #Singles
        if(detgen=='s'):
          slist=np.zeros(0)
          while(slist.size==0):
            spin=np.random.randint(2)
            ospin=np.mod(spin+1,2)
            rlist=np.arange(act[0],1+occ[state][spin])
            slist=np.arange(1+occ[state][spin],14+1)

          ospin_list=[str(x) for x in range(1,1+occ[state][ospin])]
          r=sorted(np.random.choice(rlist,size=occ[state][spin]-act[0],replace=False))
          s=np.random.choice(slist,size=1)
          core=[str(x) for x in range(1,act[0])]
          r=[str(x) for x in list(r)]
          s=[str(x) for x in list(s)]
          spin_list=list(core)+list(r)+list(s)

          if(ospin<spin): detstring+='  '+' '.join(ospin_list)+'\n'+'  '+' '.join(spin_list)+'\n\n'
          else: detstring+='  '+' '.join(spin_list)+'\n'+'  '+' '.join(ospin_list)+'\n\n'
        #Full active space
        elif(detgen=='a'):
          acts=np.arange(act[0],act[1]+1)
          r=sorted(np.random.choice(acts,size=occ[state][0]-act[0]+1,replace=False))
          s=sorted(np.random.choice(acts,size=occ[state][1]-act[0]+1,replace=False))
          core=[str(x) for x in range(1,act[0])]
          r=[str(x) for x in list(r)]
          s=[str(x) for x in list(s)]
          detstring+='  '+' '.join(core+r)+'\n'+'  '+' '.join(core+s)+'\n\n'
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
  N=25
  Ndet=10
  gsw=0.8
  geninput(detgen,N,Ndet,gsw,basename='run1a')
