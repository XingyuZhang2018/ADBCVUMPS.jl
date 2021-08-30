#!/bin/bash
model=Kitaev_Heisenberg{Float64}
D=4
chi=50
atype=CuArray
degree_right=266.0
degree_wrong=265.0

cd ~/../../data/xyzhang/ADBCVUMPS/Kitaev_Heisenberg/

cp -r ${model}\(${degree_right}\)_${atype} ${model}\(${degree_wrong}\)_${atype}

rm ${model}\(${degree_wrong}\)_${atype}/${model}\(${degree_right}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.log

mv ${model}\(${degree_wrong}\)_${atype}/${model}\(${degree_right}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.jld2 ${model}\(${degree_wrong}\)_${atype}/${model}\(${degree_wrong}\)_${atype}_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter2.jld2

cd ~/research/ADBCVUMPS.jl/job/

qsub -V degree${degree_wrong}_D${D}_chi${chi}