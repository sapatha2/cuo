method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 14
    orbfile b3lyp_iao_b.orb 
    include b3lyp_iao_b.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
  }
}
include b3lyp_iao_b.sys
