#Analyze data 
import numpy as np 
import sys
sys.path.append('/Users/shiveshpathak/Box Sync/Research/Work/si_model_fitting')
##sys.path.append('/u/sciteam/sapatha2/si_model_fitting')
from analyze_jsonlog import compute_and_save
import pandas as pd 
import matplotlib.pyplot as plt 

'''
fnames=["Cuvtz0_B3LYP_s"+str(i)+"_g0."+str(j)+".vmc.json" for i in range(1,10) \
for j in range(1,10)]

compute_and_save(fnames,save_name="saved_data.csv")
'''

#Old
oldvals=[
-212.932287,
 -52.741041,
  73.781093,
 -58.540136,
  71.159426,
  60.623498,
 -45.962795,
 -55.074815,
  65.171217,
 -54.168377,
 -76.342989,
   0.247975,
  -0.346328,
   0.274964,
  -0.334073,
  -0.284947,
   0.216331,
   0.259073,
  -0.305940,
   0.254787,
   0.358009,
   0.060127,
   0.035587,
   0.008871,
   0.024957,
  -0.050461,
   0.102308,
   0.090843,
   0.026946,
   0.082695,
  -0.112122]

olderrs=[
 0.001614,
 0.111120,
 0.203582,
 0.189033,
 0.169706,
 0.124825,
 0.137861,
 0.108285,
 0.155728,
 0.122364,
 0.144339,
 0.000520,
 0.000955,
 0.000887,
 0.000795,
 0.000587,
 0.000652,
 0.000508,
 0.000729,
 0.000578,
 0.000676,
 0.003205,
 0.005620,
 0.008071,
 0.004302,
 0.003719,
 0.004125,
 0.003238,
 0.003407,
 0.003289,
 0.004414]

#New
newvals=[
-212.932522,
 -52.794652,
  73.768400,
 -58.541952,
  71.139592,
  60.689811,
 -46.065754,
 -55.132988,
  65.099452,
 -54.240342,
 -76.250190,
   0.247941,
  -0.346440,
   0.274933,
  -0.334094,
  -0.285019,
   0.216340,
   0.258923,
  -0.305728,
   0.254731,
   0.358095,
   0.000086,
   0.000105,
   0.000133,
   0.000020,
  -0.000057,
   0.000171,
   0.000055,
   0.000027,
   0.000143,
  -0.000139]

newerrs=[
  0.001482,
  0.122529,
  0.195873,
  0.178452,
  0.167374,
  0.140091,
  0.136974,
  0.118085,
  0.161319,
  0.117472,
  0.157535,
  0.000576,
  0.000920,
  0.000838,
  0.000786,
  0.000658,
  0.000644,
  0.000555,
  0.000758,
  0.000552,
  0.000739,
  0.000370,
  0.000517,
  0.000409,
  0.000493,
  0.000424,
  0.000322,
  0.000386,
  0.000453,
  0.000377,
  0.000533]


plt.subplot(221)
plt.title("Term1")
plt.errorbar(np.arange(10),oldvals[1:11],yerr=olderrs[1:11],fmt='rs',label='bootstrap')
plt.errorbar(np.arange(10),newvals[1:11],yerr=newerrs[1:11],fmt='go',label='new')
plt.ylabel("Value")
plt.xlabel("parameter")
plt.legend(loc=1)
plt.subplot(222)
plt.title("Term2")
plt.errorbar(np.arange(10),oldvals[11:21],yerr=olderrs[11:21],fmt='rs')
plt.errorbar(np.arange(10),newvals[11:21],yerr=newerrs[11:21],fmt='go')
plt.ylabel("value")
plt.xlabel("parameter")
plt.subplot(223)
plt.title("dE/dp")
plt.errorbar(np.arange(10),oldvals[21:],yerr=olderrs[21:],fmt='rs')
plt.errorbar(np.arange(10),newvals[21:],yerr=newerrs[21:],fmt='go')
plt.ylabel("value")
plt.xlabel("parameter")
plt.show()
