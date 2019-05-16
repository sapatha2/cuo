import pandas as pd
from ed import h1_moToIAO
import numpy as np 

df = pd.read_pickle('analysis/oneparm.pickle')
parms = df.iloc[4][['mo_n_4s','mo_n_2ppi','mo_n_2pz','mo_t_pi','mo_t_dz','mo_t_sz','mo_t_ds']]
e = h1_moToIAO(parms)#,printvals=True)

