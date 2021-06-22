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
model = Kitaev(-1.0,-1.0,-1.0)
bcipeps, key = init_ipeps(model; atype = Array, D=3, χ=30, tol=1e-10, maxiter=20)
res = optimiseipeps(bcipeps, key; f_tol = 1e-10, opiter = 100, verbose = true)
e = minimum(res)
@show e