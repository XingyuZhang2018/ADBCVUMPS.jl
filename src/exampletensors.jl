"""
    model_tensor(::Ising, β::Real)

return the  `MT <: HamiltonianModel` bulktensor at inverse temperature `β` for two-dimensional
square lattice tensor-network.
"""
function model_tensor(model::Ising, β::Real)
    Ni,Nj = model.Ni, model.Nj
    a = reshape(Float64[1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1], 2, 2, 2, 2)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    q = 1 / sqrt(2) * [cβ + sβ cβ - sβ; cβ - sβ cβ + sβ]
    M = ein"abcd,ai,bj,ck,dl -> ijkl"(a, q, q, q, q)
    CopyM(M, Ni, Nj)
end

"""
    CopyM(M, Ni, Nj)

copy M to a `Ni × Nj` array 
"""
function CopyM(M, Ni, Nj)
    m = Array{Array,2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        m[i,j] = M
    end
    return m
end