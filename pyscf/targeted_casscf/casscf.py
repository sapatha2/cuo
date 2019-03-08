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
basename='gs0'
chkfile="rohf/Cuvdz_r1.725_s1_ROHF_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

'''
state_id=5 #which state to calculate, 0 is GS
ncore=4
ncas=10
nelecas=(9,8)
casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.chkfile='vdz_spin1_casscf'+str(state_id)+'.chk'
casscf.frozen=[0,1,2,3] #3px, 3py, 3pz
casscf.fix_spin_(ss=0.75)
casscf.fcisolver.wfnsym='E2y'
casscf.kernel()
casscf.verbose=4
casscf.analyze()
'''

#Run CASSCF vtz
chkfile="rohf/Cuvtz_r1.725_s1_ROHF_0.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

state_id=0 #which state to calculate, 0 is GS
ncore=4
ncas=10
nelecas=(9,8) #Sz=1/2
s=3           #2*S
casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.frozen=[0,1,2,3] #3s, 3px, 3py, 3pz
casscf.fix_spin_(ss=0.5*s*(1+0.5*s))
casscf.fcisolver.wfnsym='E2y'
casscf.diis=scf.ADIIS

casscf.chkfile='vtz_'+casscf.fcisolver.wfnsym+'_spin'+str(s)+'_casscf'+str(state_id)+'.chk'

casscf.kernel()
casscf.verbose=4
casscf.analyze()

#Mirrored vtz (for IAOs)
'''
#chkfile="rohf/Cuvtz_r1.725_s1_ROHF_0.chk"
chkfile="Cuvtz_r1.725_s1_B3LYP_0mirror.chk"
mol=lib.chkfile.load_mol(chkfile)
m=ROHF(mol)
m.__dict__.update(lib.chkfile.load(chkfile,'scf'))

state_id=0 #which state to calculate, 0 is GS
ncore=4
ncas=10
nelecas=(9,8)
casscf=mcscf.CASSCF(m,ncore=ncore,ncas=ncas,nelecas=nelecas).state_specific_(state_id)
casscf.chkfile='vtz_spin1_casscf'+str(state_id)+'mirror.chk'
casscf.frozen=[0,1,2,3] #3px, 3py, 3pz
casscf.fix_spin_(ss=0.75)
casscf.fcisolver.wfnsym='E1x'
casscf.kernel()
casscf.verbose=4
casscf.analyze()
'''
