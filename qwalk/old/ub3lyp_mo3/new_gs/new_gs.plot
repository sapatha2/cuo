method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 30
    orbfile new_gs.orb 
    include new_gs.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
  }
}
include new_gs.sys
