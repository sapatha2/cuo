import pandas as pd
import numpy as np 
iao_labels=['$\epsilon_{d_{z^2}}$' ,'$\epsilon_{d_\pi}$', '$\epsilon_{d_\delta}$', 
'$\epsilon_{4s}$', '$\epsilon_{p_\pi}$', '$\epsilon_{p_z}$', '$t_\pi$', 
'$t_{dz}$', '$t_{sz}$', '$t_{ds}$', '$J_{sd}$', '$U_s$']

model_labels=[r'',
r'$ +\ \bar{c}_{d_\pi}^\dagger \bar{c}_{p_\pi}$',
r'$ +\ \bar{c}_{d_{z^2}}^\dagger \bar{c}_{p_z}$',
r'$ +\ \bar{c}_{d_\pi}^\dagger \bar{c}_{p_\pi}, \bar{c}_{4s}^\dagger \bar{c}_{p_z}$',
r'$ +\ \bar{c}_{d_\pi}^\dagger \bar{c}_{p_\pi}, \bar{c}_{d_z^2}^\dagger \bar{c}_{4s}$',
r'$ +\ \bar{c}_{d_z^2}^\dagger \bar{c}_{p_z}, \bar{c}_{d_z^2}^\dagger \bar{c}_{4s}$']

model_labels = model_labels[:3]

df = pd.read_pickle('analysis/params.pickle')
data = np.array(list(df['params_iao_mu'].values))
data_err = np.array(list(df['params_err'].values))
data = data.T
data_err = data_err.T
data = data[:,:3]
data_err = data_err[:,:3]
data = data.T
data_err = data_err.T

#IAO
my_str='\\begin{tabular}{l|llllllllllll}\n'+\
  '&'+' & '.join(iao_labels)+' \\\\ \hline \n'

for i in range(data.shape[0]):
  p = np.around(data[i],2)
  perr = np.around(data_err[i],2)
  print(p)
  print(perr)

  add_str = 'Min '+model_labels[i]
  for z in range(len(p)):
    add_str += '& '+str(p[z])+'('+str(perr[z])+')'
  my_str+=add_str + '\\\\\n'
my_str+='\\end{tabular} \\\\\n'
print(my_str)
