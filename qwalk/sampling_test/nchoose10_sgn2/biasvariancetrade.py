#Analyze data 
import numpy as np 
import sys
#sys.path.append('/Users/shiveshpathak/Box Sync/Research/Work/si_model_fitting')
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
from analyze_jsonlog import compute_and_save,compute_and_save_cut
import json

fnames=["Cuvtz0_B3LYP_s3_g0.1.vmc.json"]
ps=[1]

for i in range(len(fnames)):
  cuts=[]
  vals=[]
  errs=[]
  for cut in np.linspace(25.0,5.0,10):
    for s in range(5): #5 samples of each
      df=compute_and_save_cut([fnames[i]],cut,ps[i])
      x=df[df['deriv']=='dpenergy_'+str(ps[i])]
      print(cut,x['value'],x['err'])
      cuts.append(cut)
      vals.append(list(x['value'].values))
      errs.append(list(x['err'].values))
  d={'cuts':cuts,'vals':vals,'errs':errs}
  json.dump(d,open("bvt_"+str(fnames[i])+"_"+str(ps[i])+".json","w"))
  print(d)
