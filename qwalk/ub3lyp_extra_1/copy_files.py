import os 
import numpy as np 

for basestate in np.arange(4):
  for gsw in np.arange(0.2,1.2,0.2):
    src_dir = '~/scratch/cuo/qwalk/ub3lyp_extra_1/gsw'+str(np.around(gsw,2))+'b'+str(basestate)
    dest_dir = 'gsw'+str(np.around(gsw,2))+'b'+str(basestate)
    print(src_dir, dest_dir)
    os.system('cp '+src_dir+'/*.o ./'+(dest_dir))
    os.system('cp '+src_dir+'/*.log ./'+str(dest_dir))
