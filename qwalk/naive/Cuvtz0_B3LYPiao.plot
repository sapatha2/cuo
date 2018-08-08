method { 
plot
orbitals {
cutoff_mo
  magnify 1
  nmo 20 
  orbfile Cuvtz0_B3LYPiao.orb
  include Cuvtz0_B3LYPiao.basis
  centers { useglobal }
}
plotorbitals {
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
}
}
include Cuvtz0_B3LYP.sys
