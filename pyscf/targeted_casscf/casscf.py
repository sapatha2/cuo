#Various analyses 
import json
from pyscf import lib,gto,scf,mcscf,fci,lo,ci,cc
from pyscf.scf import ROHF,UHF,ROKS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf2qwalk import print_qwalk_mol
from pyscf.mcscf import newton_casscf
from pyscf2qwalk import print_qwalk,print_cas_slater

#ROHF vdz: -212.238775602428
#CASCI  (12,1) -> -212.23877560242857
#CASCI  (12,2) -> -212.23877560237685
#CASCI  (10,4) -> -212.23896164489912 (2, 2, 1, 1e-4, S2=0.75)
#CASCI  (5,9)  -> -212.2394888065297  (2, ...,2, 1, 1e-4, S2=0.75)

#CASSCF (12,1) -> -212.238776082138 
#CASSCF (12,2) -> -212.23877608213763
#CASSCF (10,4) -> -212.2573301543326  (With newton!)
#CASSCF (5,9)  -> -212.267225451277   (With newton!)

#ROHF vtz: -212.242957835547 
#CASSCF (12,1) -> -212.24349945678875 (With newton!)
#CASSCF (10,4) -> -212.26259298434985 (With newton, tol=1e-5)
#CASSCF (5,9)  -> -212.273716904284   (With newton, tol=1e-5)

#Run CASSCF
basename='gs0'
chkfile="Cuvdz_r1.725_s1_ROHF_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

state_id=0 #which state to calculate, 0 is GS
ncore=4
ncas=10
nelecas=(9,8)
#casscf=newton_casscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.chkfile='vdz_spin3_casscf'+str(state_id)+'.chk'
casscf.frozen=[0,1,2,3] #3px, 3py, 3pz
#casscf.fix_spin_(shift=-1.5,ss=0.75)
casscf.fix_spin_(ss=3.75)
casscf.fcisolver.wfnsym='E1y'
casscf.kernel()
casscf.verbose=4
casscf.analyze()

#casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas)
#mo=casscf.sort_mo([4,5,6,7,8,9,10,11,12,13])
#casscf.kernel(mo)

#casscf=newton_casscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas)
#casscf=casscf.fix_spin(ss=0.75)
#casscf.conv_tol=1e-5
#casscf.chkfile=basename+'.chk'
#casscf.fcisolver.wfnsym='E1y'
#print(casscf.kernel()[0])

#Analyze CASSCF
#print_qwalk(mol,m,basename=basename)
#print_cas_slater(casscf,basename+'.orb', basename+'.basis',open(basename+'slater','w'),1e-15,open(basename+'.json','w'))
