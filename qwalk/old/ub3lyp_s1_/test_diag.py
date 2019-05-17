import pandas as pd 
import numpy as np 

e = pd.read_pickle('analysis/avg_eig.pickle')
p = pd.read_pickle('analysis/oneparm.pickle')
model = 0
var = ['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz','mo_t_sz','mo_t_ds','Jsd','Us']

e = e[e['model']==model]
e['energy'] -= min(e['energy'])
p = p.iloc[model][var]

e_p = np.dot(e[var],p)
ind = np.argsort(e_p)
print(e[['Sz']+var].iloc[ind])
