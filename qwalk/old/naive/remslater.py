#Remove all generated slater files
import sys
from os import remove 
import os.path

N=0
if(len(sys.argv)==1):
  N=1000
  print("Removing all files")
else:   
  N=int(sys.argv[1])
  print("Removing all files up to index "+str(N))

el='Cu'
charge=0
for method in ['ROHF','B3LYP','PBE0']:
  for basis in ['vdz','vtz']:
    for i in range(1,N+1):
      basename=el+basis+str(charge)+"_"+method
      fname=basename+".slater"+str(i)
      
      if(os.path.isfile(fname)):
        remove(fname)
