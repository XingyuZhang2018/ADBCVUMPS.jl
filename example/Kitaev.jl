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
folder = "E:/1 - research/4.9 - AutoDiff/data/ADBCVUMPS.jl/Kitaev_complex/"
bulk, key = init_ipeps(Kitaev(-1.0, -1.0, -1.0);folder=folder, atype = Array, D=2, χ=20, tol=1e-10, maxiter=10, miniter=2)
optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 100, verbose = true)