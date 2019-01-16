method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 30
    orbfile gs5.orb 
    include gs5.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
  }
}
include gs5.sys
