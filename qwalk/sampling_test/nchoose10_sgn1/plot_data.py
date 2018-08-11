#Plot all data from collect_data.json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_json("collect_data.json")
sns.set_style("ticks")


#Energy
plt.title("VMC Energy")
E0=-213.361501
err0=0.001006581014
df['E']-=E0
df['E']*=27.2
df['Err']*=27.2
g=sns.FacetGrid(hue='S',data=df,sharey=False)
g.map(plt.errorbar,'G','E','Err',marker='o')
plt.show()

#Parameter derivatives
plt.title("Parameter derivatives")
g=sns.FacetGrid(hue='S',col='Nd',data=df,sharey=False)
g.map(plt.errorbar,'G','Pd','Pderr',marker='o')
plt.show()

#Energy derivatives
'''
df['Ed']=2*df['Ed']-df['E']*df['Pd']
#df['Ederr']= #THIS IS THE HARD THING TO DO
g=sns.FacetGrid(hue='S',col='Nd',data=df,sharey=False)
g.map(plt.errorbar,'G','Ed','Ederr',marker='o')
plt.show()
'''
