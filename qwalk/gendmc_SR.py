#DMC files only for energy and small RDM (not full basis elements)
import numpy as np

######################################################################
#USER OPTIONS, ONLY THING TO EDIT
cutoff="0p2" #cutoff for slater
N=20 #number of expansions we want, minimum 1
nblock=100 #number of vmc blocks we want, minimum 1
timestep=0.01 #timestep, minimum 0
nodes=20 #number of nodes (32 ppn)
walltime="01:00:00"
######################################################################

el='Cu'
charge=0
minao={}

#Read in template file 
i=0
edit_ind=[] #Indices of lines to edit
template=[]
with open("dmc_template_SR","r") as f:
  for line in f:
    template.append(line)
    if("##" in line):
      edit_ind.append(i)
    i+=1
f.close()

#Write out dmc files
#for method in ['ROHF','B3LYP','PBE0']:
#  for basis in ['vdz','vtz']:
for method in ['B3LYP']:
  for basis in ['vtz']:
    for i in range(1,N+1):
      basename=el+basis+str(charge)+"_"+method
      
      template[edit_ind[0]]="  nblock "+str(nblock)+"\n"
      template[edit_ind[1]]="  timestep "+str(timestep)+"\n"
      template[edit_ind[2]]="include "+basename+".sys"
      template[edit_ind[3]]="  wf1 { include "+basename+".slater"+str(i)+cutoff+" }"+"\n"
      
      fname=basename+".dmc_E_SR_"+str(i)+cutoff
      fout=open(fname,"w")
      fout.write("".join(template))
      fout.close()

#Write dmc pbs files
for method in ['B3LYP']:
  for basis in ['vtz']:
    for i in range(1,N+1):
      basename=el+basis+str(charge)+"_"+method+".dmc_E_SR_"+str(i)+cutoff
      cpypath="/u/sciteam/$USER/cuo/qwalk/"+el+basis+str(charge)+"_"+method

      contents="#!/bin/bash \n"+\
      "#PBS -q normal\n"+\
      "#PBS -l nodes="+str(nodes)+":ppn=32:xe \n"+\
      "#PBS -l walltime="+walltime+"\n"+\
      "#PBS -N "+basename+"\n"+\
      "#PBS -e "+basename+".perr \n"+\
      "#PBS -o "+basename+".pout \n"+\
      "mkdir -p /scratch/sciteam/$USER/"+basename+"\n"+\
      "cd /scratch/sciteam/$USER/"+basename+"\n"+\
      "cp "+cpypath+".dmc_E_SR_"+str(i)+cutoff+" .\n"+\
      "cp "+cpypath+".slater"+str(i)+cutoff+" .\n"+\
      "cp "+cpypath+".optjast3 .\n"+\
      "cp "+cpypath+".sys .\n"+\
      "cp "+cpypath+".orb .\n"+\
      "cp "+cpypath+".basis .\n"+\
      "cp "+cpypath+"iao.orb .\n"+\
      "cp "+cpypath+"iao.basis .\n"+\
      "aprun -n "+str(nodes*32)+" /u/sciteam/$USER/mainline/bin/qwalk "+basename+" &> "+basename+".out\n"

      fname=basename+".pbs"
      fout=open(fname,"w")
      fout.write(contents)
      fout.close()


