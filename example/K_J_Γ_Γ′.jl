using ADBCVUMPS
using BCVUMPS
using CUDA
using Random
using Test
using OMEinsum
using Optim
using Zygote
CUDA.allowscalar(false)

Random.seed!(100)
device!(3)
folder = "./../../../../data1/xyzhang/ADBCVUMPS/K_J_Γ_Γ′_1x2/"
bulk, key = init_ipeps(K_J_Γ_Γ′(-1.0, 0.0, 0.03, 0.0);folder=folder, atype = CuArray, D=5, χ=120, tol=1e-10, maxiter=10, miniter=2)
optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 1000, verbose = true)