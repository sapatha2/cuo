import json 
import pandas as pd
import numpy as np 
from pyscf import lib,gto,scf,mcscf, fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS

#chkfile="Cuvtz_r1.725_l1_s1_B3LYP_86.chk"
#chkfile='Cuvtz_r1.725_l2_s1_B3LYP_1.chk'
#chkfile='Cuvtz_r1.725_l2_s3_B3LYP_10.chk'
#chkfile='Cuvtz_r1.725_l2_s3_B3LYP_18.chk'
chkfile='Cuvtz_r1.725_l0_s1_B3LYP_23.chk'
#chkfile='Cuvtz_r1.725_l4_s3_B3LYP_22.chk'
#chkfile='Cuvtz_r1.725_l2_s1_B3LYP_24.chk'
#chkfile='Cuvtz_r1.725_l2_s1_B3LYP_25.chk'
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
print(m.analyze())

#df=pd.read_csv("cuo.csv")
#df=df[df['basis']=='vtz']
#df=df[df['bond-length']==1.725]
#print(df.sort_values(by=['E']))
