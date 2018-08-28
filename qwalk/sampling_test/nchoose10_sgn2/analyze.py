#Analyze data 
import numpy as np 
import sys
#sys.path.append('/Users/shiveshpathak/Box Sync/Research/Work/si_model_fitting')
sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
from analyze_jsonlog import compute_and_save

fnames=["Cuvtz0_B3LYP_s"+str(i)+"_g0."+str(j)+".vmc.json" for i in range(1,10) \
for j in range(1,10)]

compute_and_save(fnames,save_name="saved_data.csv")
