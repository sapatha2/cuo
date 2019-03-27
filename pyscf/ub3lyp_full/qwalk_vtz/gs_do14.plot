method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 147
    orbfile gs_do14.orb 
    include gs_do14.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10 
    11    12    13    14   
    
    134 135 136 137 138 139 140 141 142 143 144 145 146 147
  }
}
include gs_do14.sys
