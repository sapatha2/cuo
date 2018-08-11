#Plot all data from collect_data.json

#Plot energies
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
