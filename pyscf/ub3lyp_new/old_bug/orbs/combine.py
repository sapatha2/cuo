import cubetools

f1 = cubetools.read_cube('test_mom.plot.orb7.cube')
f2 = cubetools.read_cube('test_mom.plot.orb10.cube')
f1 = cubetools.normalize_abs(f1)
f2 = cubetools.normalize_abs(f2)

f1['data'] -= f2['data']
cubetools.write_cube(f1,'test_mom.plot.orbDiff.cube')

f1['data'] += 2*f2['data']
cubetools.write_cube(f1,'test_mom.plot.orbSum.cube')
