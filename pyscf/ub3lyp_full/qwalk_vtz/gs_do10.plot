method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 14
    orbfile gs_do10.orb 
    include gs_do10.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
  }
}
include gs_do10.sys
