import os
import numpy as np 

for gsw in np.arange(0.1,1.1,0.1):
  for basestate in np.arange(6):
    dir='gsw'+str(np.round(gsw,2))+'b'+str(basestate)
    print(dir)
    os.system('cp -u ~/scratch/cuo/qwalk/ub3lyp_d10_s1/'+dir+'/*.log ./'+dir)
    os.system('cp -u ~/scratch/cuo/qwalk/ub3lyp_d10_s1/'+dir+'/*.o ./'+dir)
