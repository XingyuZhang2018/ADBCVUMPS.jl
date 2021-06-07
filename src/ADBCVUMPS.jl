module ADBCVUMPS

using Zygote
using OMEinsum
using BCVUMPS
using BCVUMPS: HamiltonianModel
using BCVUMPS: _mattype, _arraytype

export num_grad
export TFIsing, Heisenberg
export hamiltonian, HamiltonianModel
export init_ipeps, energy, optimiseipeps

include("hamiltonianmodels.jl")
include("autodiff.jl")
include("bcipeps.jl")
include("variationalipeps.jl")

end
