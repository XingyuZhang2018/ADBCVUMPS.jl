#!/bin/bash
model=K_Γ{Float64}
D=4
chi=20

# cd ../data/SL/ 

# for i in $(seq 0.0 5.0 90.0); do cp ${model}\(-0.0\)_Array ${model}\(-${i}\)_Array -r; done

# for i in $(seq 0.0 5.0 90.0); do cd ${model}\(-${i}\)_Array && rm ${model}\(-0.0\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.log && mv ${model}\(-0.0\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld${D} ${model}\(-${i}\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld${D} && cd ..; done

# cd ../../job/

for i in $(seq 0.1 0.05 1.0); do cp K_gamma0.1_D${D}_chi${chi} K_gamma${i}_D${D}_chi${chi} && sed -i "7s/0.1/$i/1" K_gamma${i}_D${D}_chi${chi}; done

# grep degree *_chi${chi}
# cd ~/../../data/xyzhang/ADBCVUMPS/K_Γ/
# for i in $(seq 0.01 0.01 0.08); do cd ${model}\(${i}\)_CuArray && cp D4_chi20_tol1.0e-10_maxiter10_miniter2.jld2 D4_chi50_tol1.0e-10_maxiter10_miniter2.jld2 && cd ..; done
for i in $(seq 0.1 0.05 1.0); do qsub -V K_gamma${i}_D${D}_chi${chi}; done
for i in $(seq 0.15 0.05 1.0); do rm K_gamma${i}_D${D}_chi${chi}; done
# for i in $(seq 72.0 3.0 90.0); do qdel $i; done
# SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
# echo $SHELL_FOLDER