method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 30
    orbfile gs1.orb 
    include gs1.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
  }
}
include gs1.sys
