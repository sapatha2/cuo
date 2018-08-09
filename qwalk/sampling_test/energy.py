import json 

'''
E=[]
Err=[]
G=[]
S=[]
sgn=[]
nchoose=[]

for s in range(1,10):
  for g in range(1,10):
    with open("/u/sciteam/sapatha2/scratch/nchoose10_sgn0/Cuvtz0_B3LYP_s"+str(s)+"_g0."+str(g)+".vmc.o","r") as f:
      for line in f:
        if("total_energy0" in line):
          sp1=line.split("+/-")
          E.append(float(sp1[0].split(" ")[-2]))
          Err.append(float(sp1[1].split("(")[0]))
          G.append(g)
          S.append(s)
    f.close()

sgn=[0]*len(E)
nchoose=[10]*len(E)

d={'E':E,'Err':Err,'G':G,'S':S,'sgn':sgn,'nchoose':nchoose}
json.dump(d,open("energy.json","w"))
'''

d=json.load(open("energy.json","w"))

