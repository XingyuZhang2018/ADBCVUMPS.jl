#!/bin/bash
model=Kitaev_Heisenberg{Float64}
D=3
chi=20

degree_right=85.0
degree_wrong=80.0

cd ../data/all/

cp ${model}\(${degree_right}\)_Array/${model}\(${degree_right}\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld2 ${model}\(${degree_wrong}\)_Array/${model}\(${degree_wrong}\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.jld2

cp ${model}\(${degree_right}\)_Array/up_D$[$D**2]_chi${chi}.jld2 ${model}\(${degree_wrong}\)_Array/up_D$[$D**2]_chi${chi}.jld2

cp ${model}\(${degree_right}\)_Array/down_D$[$D**2]_chi${chi}.jld2 ${model}\(${degree_wrong}\)_Array/down_D$[$D**2]_chi${chi}.jld2

rm ${model}\(${degree_wrong}\)_Array/${model}\(${degree_wrong}\)_Array_D${D}_chi${chi}_tol1.0e-10_maxiter10_miniter5.log

cd ../../job/

qsub -V degree${degree_wrong}_D${D}_chi${chi}