#Generation for sampling files
#Loop nchoose 
#  Loop sign
#    Loop nsample
#      Loop GS weights 
#        DMC

import numpy as np 
import random 
import os 

nchooses=[10] #subset of determinants to choose 
nsigns=1 #number of distinct sign choices
nsamples=10 #number of samples of nchoose determinants

nup=8 #number of up electrons
ndn=7 #number of down electrons
actmin=6 #active space minimum
actmax=14 #active space maximum
nblock_vmc=1  #number of vmc blocks
nodes=1 #nodes to use 
walltime='01:00:00' #walltime

#VMC input file template
i=0
vmc_edit_ind=[] #Indices of lines to edit
vmc_template=[]
with open("vmc_template","r") as f:
  for line in f:
    vmc_template.append(line)
    if("##" in line):
      vmc_edit_ind.append(i)
    i+=1
f.close()


for nchoose in nchooses:
  for i in range(nsigns):
    sign=[1]+(2*np.random.randint(0,2,nchoose)-1).tolist() #sign choice
    
    #Make working directory
    ind=i
    directory="./nchoose"+str(nchoose)+"_sgn"+str(ind)
    while(os.path.exists(directory)):
      ind+=1
      directory="./nchoose"+str(nchoose)+"_sgn"+str(ind)
    os.makedirs(directory)
    os.system("cp ./Cuvtz0_B3LYP.* "+directory)         #Copy relevant files
    os.system("cp ./Cuvtz0_B3LYPiao.* "+directory)

    for nsample in range(nsamples):
      up=[list(range(1,actmin))+random.sample(range(actmin,actmax+1), nup) for j in range(nchoose)] #determinant choices up
      dn=[list(range(1,actmin))+random.sample(range(actmin,actmax+1), ndn) for j in range(nchoose)] #determinant choices down
      assert(len(set(tuple(k) for k in up))==nchoose) #make sure you don't have repeats
      assert(len(set(tuple(k) for k in dn))==nchoose) #make sure you don't have repeats
      
      for gsw in np.arange(0.1,1.0,0.1):
        gsw=float("{0:.2f}".format(gsw)) #round to 1 decimal point
        w=np.array(sign[:]).astype(float) #determinant weights
        w[0]*=(gsw**0.5)
        w[1:]*=((1-gsw)/nchoose)**0.5
        assert((np.dot(w,w)-1.0)<1e-12) #make sure it's normalized
        assert(w[0]==gsw**0.5)          #make sure you have the right first element

        #Generate slater files
        slatf="Cuvtz0_B3LYP_s"+str(nsample)+"_g"+str(gsw)+".slater"
        f=open(directory+"/"+slatf,"w")
        w=[str(x) for x in w]
        up=[[str(y) for y in x] for x in up]
        dn=[[str(y) for y in x] for x in dn]
        
        states=""
        for p in range(nchoose):
          states+=" ".join(up[p])+"\n"+" ".join(dn[p])+"\n\n"
        print(states)

        content=\
        "SLATER\n"+\
        "ORBITALS  {\n"+\
        "CUTOFF_MO\n"+\
        "MAGNIFY 1.0\n"+\
        "NMO 14\n"+\
        "ORBFILE Cuvtz0_B3LYP.orb\n"+\
        "INCLUDE Cuvtz0_B3LYP.basis\n"+\
        "CENTERS { USEGLOBAL }\n"+\
        "}\n"+\
        "DETWT { "+" ".join(w)+" }\n"+\
        "STATES {\n"+states+\
        "}\n"
        f.write(content)

        #Generate VMC files
        vmc_template[vmc_edit_ind[0]]="  nblock "+str(nblock_vmc)+"\n"
        vmc_template[vmc_edit_ind[1]]="  wf1 { include "+slatf+" }"+"\n"

        vmcf="Cuvtz0_B3LYP_s"+str(nsample)+"_g"+str(gsw)+".vmc"
        f=open(directory+"/"+vmcf,"w")
        f.write("".join(vmc_template))
        f.close()

        #Generate QSUB files
        cpypath="/u/sciteam/$USER/cuo/qwalk/sampling_test/"+directory
        contents="#!/bin/bash \n"+\
        "#PBS -q normal\n"+\
        "#PBS -l nodes="+str(nodes)+":ppn=32:xe \n"+\
        "#PBS -l walltime="+walltime+"\n"+\
        "#PBS -N "+vmcf+"\n"+\
        "#PBS -e "+vmcf+".perr \n"+\
        "#PBS -o "+vmcf+".pout \n"+\
        "mkdir -p /scratch/sciteam/$USER/"+directory+"\n"+\
        "cd /scratch/sciteam/$USER/"+directory+"\n"+\
        "cp "+cpypath+"/"+vmcf+" .\n"+\
        "cp "+cpypath+"/"+slatf+" .\n"+\
        "cp "+cpypath+"/Cuvtz0_B3LYP.optjast3 .\n"+\
        "cp "+cpypath+"/Cuvtz0_B3LYP.sys .\n"+\
        "cp "+cpypath+"/Cuvtz0_B3LYP.orb .\n"+\
        "cp "+cpypath+"/Cuvtz0_B3LYP.basis .\n"+\
        "cp "+cpypath+"/Cuvtz0_B3LYPiao.orb .\n"+\
        "cp "+cpypath+"/Cuvtz0_B3LYPiao.basis .\n"+\
        "aprun -n "+str(nodes*32)+" /u/sciteam/$USER/mainline/bin/qwalk "+vmcf+" &> "+vmcf+".out\n"
 
        fname=directory+"/"+vmcf+".pbs"
        fout=open(fname,"w")
        fout.write(contents)
        fout.close()
 
