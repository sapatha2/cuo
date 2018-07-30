#Generates multi slater expansions for our calculations
import numpy as np

N=2 #number of expansions we want, minimum 1
nblock=640 #number of vmc blocks we want, minimum 1
assert(N>=1)
assert(nblock>=1)

el='Cu'
charge=0
minao={}

#Read in template file 
i=0
edit_ind=[] #Indices of lines to edit
template=[]
with open("vmc_template","r") as f:
  for line in f:
    template.append(line)
    if("##" in line):
      edit_ind.append(i)
    i+=1
f.close()

#Write out slater files
#for method in ['ROHF','B3LYP','PBE0']:
#  for basis in ['vdz','vtz']:
for method in ['B3LYP']:
  for basis in ['vtz']:
    for i in range(1,N+1):
      basename=el+basis+str(charge)+"_"+method
      
      template[edit_ind[0]]="  nblock "+str(nblock)+"\n"
      template[edit_ind[1]]="include "+basename+".sys"+"\n"
      template[edit_ind[2]]="  wf1 { include "+basename+".slater"+str(i)+" }"+"\n"
      
      fname=basename+".vmc"+str(i)
      fout=open(fname,"w")
      fout.write("".join(template))
      fout.close()
