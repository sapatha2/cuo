method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 15
    orbfile qw.orb 
    include qw.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    6 7 8 11    12    13    14    
  }
}
include qw.sys 

