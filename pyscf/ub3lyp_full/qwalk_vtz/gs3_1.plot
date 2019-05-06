method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 148 
    orbfile gs3_1.orb 
    include gs3_1.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
  1 2 3 4 5 6 7 8 9 10 11 12 13 14
  #Spin down orbitals 
  134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 
  }
}
include gs3_1.sys
