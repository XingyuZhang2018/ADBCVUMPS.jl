using ADBCVUMPS
using BCVUMPS
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

Random.seed!(99)
# device!(7)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADBCVUMPS/K_J_Γ_Γ′_1x2/"
bulk, key = init_ipeps(K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02), [1.0,1.0,0.963424], 0.1;folder=folder, type = "_random", atype = Array, D=2, χ=20, tol=1e-10, maxiter=10, miniter=2)
optimiseipeps(bulk, key; f_tol = 1e-10, opiter = 0, verbose = true)
