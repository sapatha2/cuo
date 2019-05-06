#Collect all data into one place 

import json 

G=[]   #Ground state weight 
S=[]   #Sample
E=[]   #Total energy
Err=[] #Error bar
Pd=[]  #Parameter derivatives
Pderr=[] #Parameter derivative errors
Ed=[]  #Energy derivatives
Ederr=[] #Energy derivative errors
Nd=[]  #Derivative number

for s in range(1,10):
  for g in range(1,10):
    with open("Cuvtz0_B3LYP_s"+str(s)+"_g0."+str(g)+".vmc.data","r") as f:
      for line in f:
        if("total_energy0" in line):
          sp1=line.split("+/-")
          for i in range(10):
            E.append(float(sp1[0].split(" ")[-2]))
            Err.append(float(sp1[1].split("(")[0]))
            Nd.append(i)
            G.append(g)
            S.append(s)
        if("Wave function derivatives" in line):
          pd=[]
          pderr=[]
          for i in range(10):
            l=next(f).split("+/-")
            Pd.append(float(l[0]))
            Pderr.append(float(l[1]))
        if("Derivative of the energy" in line):
          ed=[]
          ederr=[]
          for i in range(10):
            l=next(f).split("+/-")
            Ed.append(float(l[0]))
            Ederr.append(float(l[1]))
    f.close()

print(len(E),len(Err),len(G),len(S),len(Pd),len(Pderr),len(Ed),len(Ederr))
d={'Nd':Nd,'E':E,'Err':Err,'G':G,'S':S,'Pd':Pd,'Pderr':Pderr,'Ed':Ed,'Ederr':Ederr}
json.dump(d,open("collect_data.json","w"))

'''
import pandas as pd
import matplotlib.pyplot as plt

d=json.load(open("energy.json","r"))
df=pd.DataFrame(d)

#Ground state
E0=-213.361501
err0=0.001006581014
df['E']-=E0
df['E']*=27.2
df['Err']*=27.2

for S in range(max(d['S'])):
  ind=(df['S']==S)
  plt.errorbar(df['G'][ind]*0.1,df['E'][ind],yerr=df['Err'][ind],marker='o')

plt.ylabel("E(VMC) - E0(VMC), eV")
plt.xlabel("Ground state weight")
plt.show()
'''
