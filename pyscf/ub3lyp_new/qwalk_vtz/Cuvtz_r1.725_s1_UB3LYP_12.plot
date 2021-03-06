method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 148 
    orbfile Cuvtz_r1.725_s1_UB3LYP_12.orb 
    include Cuvtz_r1.725_s1_UB3LYP_12.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
  1 2 3 4 5 6 7 8 9 10 11 12 13 14
  #Spin down orbitals 
  134 135 136 137 138 139 140 141 142 143 144 145 146 147 148
  }
}
include Cuvtz_r1.725_s1_UB3LYP_12.sys
