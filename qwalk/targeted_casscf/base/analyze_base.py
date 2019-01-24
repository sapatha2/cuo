import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_pickle('base_gosling.pickle')
df['E']-=min(df['E'])
df['E']*=27.2114
df=df.sort_by(column=['E'])
print(df[['E','n_3d','n_4s','n_2ppi','n_2pz','J_4s_3d']])
