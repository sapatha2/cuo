method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 30
    orbfile gs0.orb 
    include gs0.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
  }
}
include gs0.sys
