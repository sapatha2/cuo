method {
  plot
  orbitals {
  cutoff_mo
    magnify 1
    nmo 30
    orbfile Cuvtz_r1.963925_c0_s3_B3LYP_4Phi.orb 
    include Cuvtz_r1.963925_c0_s3_B3LYP_4Phi.basis
    centers { useglobal }
  }
  PLOTORBITALS { 
    1     2     3     4     5     6     7     8     9    10  
    11    12    13    14    15    16    17    18    19    20  
    21    22    23    24    25    26    27    28    29    30  
  }
}
include Cuvtz_r1.963925_c0_s3_B3LYP_4Phi.sys 

