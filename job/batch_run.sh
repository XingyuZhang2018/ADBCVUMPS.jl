#!/bin/bash
model=Kitaev_Heisenberg{Float64}
D=4
chi=20

# cd ../data/SL/ 

# for i in $(seq 0.0 5.0 90.0); do cp ${model}\(-0.0\)_Array ${model}\(-${i}\)_Array -r; done

# for i in $(seq 0.0 5.0 90.0); do cd ${model}\(-${i}\)_Array && rm ${model}\(-0.0\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.log && mv ${model}\(-0.0\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld${D} ${model}\(-${i}\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld${D} && cd ..; done

# cd ../../job/

for i in $(seq 277.5 2.5 290.0); do cp degree0.0_D${D}_chi${chi} degree${i}_D${D}_chi${chi} && sed -i "7s/0.0/$i/1" degree${i}_D${D}_chi${chi}; done

# grep degree *_chi${chi}

# for i in $(seq 265.0 1.0 275.0); do qsub -V degree${i}_D${D}_chi${chi}; done

# for i in $(seq 72.0 3.0 90.0); do qdel $i; done
# SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
# echo $SHELL_FOLDER