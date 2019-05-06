import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_pickle('base_gosling.pickle')
df['energy']-=min(df['energy'])
df=df.sort_values(by=['energy'])
print(df[['base_state','energy','n_3d','n_4s','n_2ppi','n_2pz','J_4s_3d']])
