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
model = Heisenberg(1,1,1.0,1.0,1.0)
bulk, key = init_ipeps(model; atype = CuArray, D=6, χ=80, tol=1e-10, maxiter=10)
res = optimiseipeps(bulk, key; f_tol = 1e-20, opiter = 1000, verbose = true)
e = minimum(res)
@show e