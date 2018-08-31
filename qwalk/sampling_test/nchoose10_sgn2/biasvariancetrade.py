#Analyze data 
import numpy as np 
import sys
import json 
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
import pandas as pd
from analyze_jsonlog import compute_and_save,compute_and_save_cut
import matplotlib.pyplot as plt

'''
#Identify large err(d<H>/dp) states
err_cut=0.15 #Error cutoff, include data points with errors bigger than this
df=pd.read_csv("saved_data.csv")
new_df=pd.read_csv("saved_data.csv")
df=df.sort_values(by=["err"],ascending=False)[df['err']>=0.015]
fnames=df['filename'].values
ps=df['deriv'].values
ps=[int(x.split("_")[1]) for x in ps]

#Cutoff analysis 
for i in range(len(fnames)):
  dat=(df.iloc[[i]])
  err=dat['err'].values
  val=dat['value'].values
  cut=10.0
  #Get new estimator
  while(err>=0.015 and cut>1e-3):
    cut/=4
    x=compute_and_save_cut([fnames[i]],cut,ps[i])
    res=x[x['deriv']==df['deriv'].values[i]]
    val=res['value'].values[0]
    err=res['err'].values[0]
    print(fnames[i],cut,val,err)
  
  if(cut>1e-3):
    #Put back into data frame
    ind=np.argsort(-new_df['err'])[i]
    new_df.at[ind,"value"]=val
    new_df.at[ind,"err"]=err
    print("##################################################")
    print(val,err)
  
#Write to CSV
new_df.to_csv("new_saved_data.csv")
'''

#Plotting
df=pd.read_csv("saved_data.csv")
new_df=pd.read_csv("new_saved_data.csv")
ind=np.argsort(-df['err'])
ind=ind[df['err']>=0.015].values

new_df=new_df.iloc[ind]
#for i in range(len(new_df)):
for i in range(1): 
  fname=new_df.iloc[i]['filename']
  deriv=new_df.iloc[i]['deriv']
  tmp=df[df['filename']==fname]
  tmp=df[df['deriv']==deriv]
  f=plt.figure()
  plt.errorbar(tmp['gsw'],tmp['value'],yerr=tmp['err'])
  plt.errorbar(new_df.iloc[i]['gsw'],new_df.iloc[i]['value'],yerr=new_df.iloc[i]['err'])
  f.savefig("tmp.pdf",bbox_inches='tight')
