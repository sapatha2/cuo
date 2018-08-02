#Generates multi slater expansions for our calculations
import numpy as np

cutoff="0p4"
wmax=0.4 #maximum magnitude of weights
Ndet=23 #number of determinants in expansion
N=20 #number of expansions we want, minimum 1
assert(N>=1)

el='Cu'
charge=0
minao={}

#Read in template file 
i=0
edit_ind=[] #Indices of lines to edit
template=[]
with open("slat_template","r") as f:
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
      
      w=2*wmax*np.random.rand(Ndet-1)-wmax
      weights=[1.0]+w.tolist()
      weights=[str(x) for x in weights]
      
      for p in np.arange(0,Ndet+5,5):
        weights.insert(p,"\n")
      
      template[edit_ind[0]]="  ORBFILE "+basename+".orb"+"\n"
      template[edit_ind[1]]="  INCLUDE "+basename+".basis"+"\n"
      template[edit_ind[2]]="DETWT { "+" ".join(weights)+" }"+"\n"
      
      fname=basename+".slater"+str(i)+cutoff
      fout=open(fname,"w")
      fout.write("".join(template))
      fout.close()
