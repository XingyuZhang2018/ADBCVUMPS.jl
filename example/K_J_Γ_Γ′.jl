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
K = -1.0
J = -0.1 * K
Γ = 0.3 * K
Γ′= -0.02 * K
folder = "./../../../../data1/xyzhang/ADBCVUMPS/K_J_Γ_Γ′/"
bulk, key = init_ipeps(K_J_Γ_Γ′(K, J, Γ, Γ′);folder=folder, atype = Array, D=2, χ=20, tol=1e-10, maxiter=10, miniter=2)
optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 100, verbose = true)