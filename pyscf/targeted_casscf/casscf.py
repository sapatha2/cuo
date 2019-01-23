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

#Run CASSCF vdz
'''
basename='gs0'
chkfile="rohf/Cuvdz_r1.725_s1_ROHF_0.chk"
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
'''

#Run CASSCF vtz
'''
#S=1,sigma0,pi0,pi1,pi2,delta5
#S=3,sigma0,pi0
chkfile="rohf/Cuvtz_r1.725_s1_ROHF_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

state_id=0 #which state to calculate, 0 is GS
ncore=4
ncas=10
nelecas=(9,8)
casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.chkfile='vtz_spin3_casscf'+str(state_id)+'.chk'
casscf.frozen=[0,1,2,3] #3px, 3py, 3pz
casscf.fix_spin_(ss=3.75)
casscf.fcisolver.wfnsym='E1y'
casscf.kernel()
casscf.verbose=4
casscf.analyze()
'''

#Mirrored vtz (for IAOs)
chkfile="rohf/Cuvtz_r1.725_s1_ROHF_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

state_id=5 #which state to calculate, 0 is GS
ncore=4
ncas=10
nelecas=(9,8)
casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.chkfile='vtz_spin1_casscf'+str(state_id)+'mirror.chk'
casscf.frozen=[0,1,2,3] #3px, 3py, 3pz
casscf.fix_spin_(ss=0.75)
casscf.fcisolver.wfnsym='E2x'
casscf.kernel()
casscf.verbose=4
casscf.analyze()
