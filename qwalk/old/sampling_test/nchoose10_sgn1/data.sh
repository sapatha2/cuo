#!/bin/bash
for S in {0..9}; do
  for G in {1..9}; do
    ~/mainline/bin/gosling "Cuvtz0_B3LYP_s"$S"_g0."$G".vmc.log" > "Cuvtz0_B3LYP_s"$S"_g0."$G".vmc.data"
  done
done
