#Analyze data 
from covariance import gather_json,test_denergy_err

jsonfn='Cuvtz0_B3LYP_s1_g0.9.vmc.json'

#Calculate errors
#test_denergy_err(jsonfn)

#Correlation matrix
df=gather_json(jsonfn)
cov=df.cov()
corr=df.corr()
