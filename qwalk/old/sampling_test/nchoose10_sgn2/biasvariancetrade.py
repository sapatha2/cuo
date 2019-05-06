#Analyze data 
import numpy as np 
import sys
import json 
#sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
sys.path.append('/Users/shiveshpathak/Box Sync/Research/Work/si_model_fitting')
from analyze_jsonlog import gather_json_df,compute_and_save
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.integrate import quad 
from scipy import interpolate 
from sklearn.metrics import r2_score

#######################################################################################################
#ESTIMATION OF LARGE ERROR BAR POINTS BY DROPPING BLOCKS
#######################################################################################################
#WRITE CSV WITH DROPPED BLOCKS
'''
err_cut=0.025 #Error cutoff, include data points with errors bigger than this
an_cut=0.001
df=pd.read_csv("saved_data.csv")
new_df=pd.read_csv("saved_data.csv")
df=df.sort_values(by=["err"],ascending=False)[df['err']>=err_cut]
fnames=df['filename'].values
ps=df['deriv'].values
ps=[int(x.split("_")[1]) for x in ps]
print("Number of derivatives to correct: "+str(df.shape[0]))

#Cutoff analysis 
for i in range(len(fnames)):
  dat=(df.iloc[[i]])
  err=dat['err'].values
  val=dat['value'].values
  cut=30.0

  new_value=[]
  new_err=[]
  new_filename=[]
  new_deriv=[]
  new_gsw=[]
  #Get new estimator
  for j in range(11):
    x=compute_and_save_cut([fnames[i]],cut,ps[i])
    res=x[x['deriv']==df['deriv'].values[i]]
    val=res['value'].values[0]
    err=res['err'].values[0]
    print(fnames[i],cut,val,err)

    #Put back into data frame
    new_value.append(val)
    new_err.append(err)
    new_filename.append(fnames[i])
    new_deriv.append("dpenergy_"+str(ps[i]))
    new_gsw.append(np.sqrt(float("0."+fnames[i].split(".")[1])))

    #Reduce cutoff 
    cut/=(30/an_cut)**(1./12.)

  d={'value':new_value,'err':new_err,'gsw':new_gsw,'deriv':new_deriv,'filename':new_filename}
  new_row=pd.DataFrame.from_dict(d)
  new_df=new_df.append(new_row,ignore_index=True)
  print(new_df)

#Write to CSV
new_df.to_csv("new_saved_data"+str(an_cut)+"m2.csv")
'''

an_cut=0.001
df=pd.read_csv("saved_data.csv")
new_df=pd.read_csv("new_saved_data"+str(an_cut)+"m2.csv")
df_cut=df.iloc[np.argsort(-df['err'].values)][df['err']>=0.025]

#PLOT SEQUENTIAL CUTOFF EFFECT
'''
j=0
for index,row in df_cut.iterrows():
  j+=1
  fname=row['filename']
  fnames=[fname.split(".")[0]+"."+str(int(x))+"."+fname.split(".")[2]+".json" for x in range(1,10)]
  deriv=row['deriv']

  new_data=new_df[new_df['filename'].isin(fnames)]
  new_data=new_data[new_data['deriv']==deriv]

  gsw=new_data['gsw'].values**2
  ind2=np.array(range(len(gsw)))[np.abs(gsw-gsw[9])<1e-3]
  for k in range(11):
    dHdp=new_data['value'].values[:]
    err=new_data['err'].values[:]
    dHdp[ind2[0]]=dHdp[9+k]
    err[ind2[0]]=err[9+k]
    dHdp=dHdp[:9]
    err=err[:9]

    plt.subplot(2,6,k+1)
    plt.errorbar(gsw[:9],dHdp,yerr=err,fmt='gs')
  plt.show()
'''

#PLOT TOTAL CUTOFF WITH POLYFITS
'''
j=0
for index,row in df_cut.iterrows():
  j+=1
  fname=row['filename']
  fnames=[fname.split(".")[0]+"."+str(int(x))+"."+fname.split(".")[2]+".json" for x in range(1,10)]
  deriv=row['deriv']
  old_data=df[df['filename'].isin(fnames)]
  old_data=old_data[old_data['deriv']==deriv]

  new_data=new_df[new_df['filename'].isin(fnames)]
  new_data=new_data[new_data['deriv']==deriv]

  plt.subplot(2,3,j)
  plt.title(str(fname.split("_")[2])+", "+str(deriv))
  plt.xlabel("GSW")
  plt.ylabel("Value")
  plt.errorbar(new_data['gsw']**2,new_data['value'],yerr=new_data['err'],fmt='gs')
  plt.errorbar(old_data['gsw']**2,old_data['value'],yerr=old_data['err'],fmt='bo-')
  
  gsw=new_data['gsw'].values**2
  ind2=np.array(range(len(gsw)))[np.abs(gsw-gsw[9])<1e-3]
  for k in range(11):
    dHdp=new_data['value'].values[:]
    dHdp[ind2[0]]=dHdp[9+k]
    dHdp=dHdp[:9]
  
    #POLYFIT
    c=np.polyfit(gsw[:9],dHdp,deg=2)
    p_interp=np.linspace(gsw[0],gsw[8],100)
    plt.plot(p_interp,c[0]*p_interp**2 + c[1]*p_interp + c[2],'k--')

  #20K BLOCK RUN
  if(j==1):
    plt.errorbar(0.1,-0.06425238737804996,yerr=0.0043890512733563095,fmt='ro')

plt.show()
'''

#LARGE CALCULATION S=3, G=0.1 DISTRIBUTION AND MEAN 
'''
df=gather_json_df('Cuvtz0_B3LYP_s3_g0.1.vmc.json')
dpsidp=df['dpwf_1'].values[:]
plt.subplot(211)
plt.hist(dpsidp,bins=200)
plt.subplot(212)
res=probplot(dpsidp,plot=plt)
plt.show()
df=compute_and_save(['Cuvtz0_B3LYP_s3_g0.1.vmc.json'],save_name='longsaved_data.csv')
'''

#######################################################################################################
#INTEGRATION OF DERIVATIVES 
#######################################################################################################

def integrate():
  coeffs=[]
  for index,row in df_cut.iterrows():
    fname=row['filename']
    fnames=[fname.split(".")[0]+"."+str(int(x))+"."+fname.split(".")[2]+".json" for x in range(1,10)]
    
    #if(fname=='Cuvtz0_B3LYP_s4_g0.4.vmc.json'):
    dHdgs=[np.zeros(9) for i in range(11)]
    sgn=[1, -1, 1, -1, -1, 1, 1, -1, 1, 1]
    
    for p in range(10):
      deriv=row['deriv'].split("_")[0]+"_"+str(p)

      old_data=df[df['filename'].isin(fnames)]
      old_data=old_data[old_data['deriv']==deriv]
      
      new_data=new_df[new_df['filename'].isin(fnames)]
      new_data=new_data[new_data['deriv']==deriv]
      
      if(deriv==row['deriv']):
        #Do procedure above
        gsw=new_data['gsw'].values**2
        ind2=np.array(range(len(gsw)))[np.abs(gsw-gsw[9])<1e-3]
        for k in range(11):
          dHdp=new_data['value'].values[:]
          dHdp[ind2[0]]=dHdp[9+k]
          dHdp=dHdp[:9]
          #plt.plot(dHdp,'o')
          #plt.show()
          dHdg=-dHdp[:]*(sgn[p]/np.sqrt(10*(1-gsw[:9])))/gsw[:9]
          dHdgs[k]+=dHdg 
      else:
        gsw=old_data['gsw'].values**2
        dHdp=old_data['value'].values[:]
        dHdg=-dHdp[:]*(sgn[p]/np.sqrt(10*(1-gsw[:9])))/gsw[:9]
        for i in range(11):
          dHdgs[i]+=dHdg

    dHdgs=np.array([list(x)+[0] for x in dHdgs])
    
    #Get H
    Hs=[np.zeros(9) for i in range(11)]
    for i in range(11):
      t=(dHdgs[i][:-1]+dHdgs[i][1:])[:]/2*-0.1
      t=np.cumsum(np.array(t)[::-1])[::-1]
      Hs[i]=t[:]

    #Plot against calculated H
    H_data=df[df['filename'].isin(fnames)]
    H_data=H_data[H_data['deriv']=='energy']
    r2scores=[]
    plt.suptitle(str(fname.split("_")[2])+", "+str(row['deriv']))
    plt.subplot(211)
    plt.xlabel("GSW")
    plt.ylabel("E-E0")
    plt.plot(H_data['gsw'].values**2,H_data['value'].values+213.361501,'o')
    for i in range(11):
      r2scores.append(r2_score(H_data['value'].values,Hs[i]-213.361501))
      plt.plot(H_data['gsw'].values**2,Hs[i],'*')
    plt.subplot(212)
    plt.xlabel("Cutoff")
    plt.ylabel("R2")
    cutoff=[30/(2.3609861416**j) for j in range(11)]
    plt.plot(cutoff, r2scores,'.')
    plt.show()

integrate()
