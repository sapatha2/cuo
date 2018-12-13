import cubetools
nup=13
ndown=12
import copy 

up=cubetools.read_cube("Cuvtz_r1.963925_c0_s1_B3LYP.plot_u.dens.cube")
down=cubetools.read_cube("Cuvtz_r1.963925_c0_s1_B3LYP.plot_d.dens.cube")
up=cubetools.normalize_abs(up)
down=cubetools.normalize_abs(down)
up['data']*=nup
down['data']*=ndown
up['data']-=down['data'] #make 'up' the spin density
cubetools.write_cube(up,"spin.dens.cube")
up['data']+=2*down['data'] #make 'up' the charge density
cubetools.write_cube(up,"chg.dens.cube")

