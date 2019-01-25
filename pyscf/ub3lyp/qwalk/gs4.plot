method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 88
    orbfile gs4.orb 
    include gs4.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13     
    77 78 79 80 81 82 83 84 85 86 87 88
  }
}
include gs4.sys
