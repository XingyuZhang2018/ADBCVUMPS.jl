module ADBCVUMPS

using BCVUMPS
using BCVUMPS: HamiltonianModel
using BCVUMPS: _mattype, _arraytype
using OMEinsum
using Zygote

export num_grad
export hamiltonian, HamiltonianModel
export TFIsing, Heisenberg, Kitaev
export init_ipeps, energy, optimiseipeps

include("hamiltonianmodels.jl")
include("autodiff.jl")
include("bcipeps.jl")
include("variationalipeps.jl")

end
