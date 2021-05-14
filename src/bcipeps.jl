using LinearAlgebra
export BCIPEPS, SquareBCIPEPS
"""
    BCIPEPS{LT<:AbstractLattice, T, N}

Infinite projected entangled pair of states.
`LT` is the type of lattice, `T` and `N` are bulk tensor element type and order.
"""
struct BCIPEPS{LT<:AbstractLattice, T, N, AT<:AbstractArray{<:AbstractArray,2}}
    bulk::AT
end
function BCIPEPS{LT}(bulk::AT) where {LT<:AbstractLattice, AT<:AbstractArray{<:AbstractArray,2}}
    T, N = eltype(bulk[1,1]), ndims(bulk[1,1])
    BCIPEPS{LT,T,N,AT}(bulk)
end

############### BCIPEPS on square lattice ###################
# size of bulk is `d × d × d × d × s`
const SquareBCIPEPS{T} = BCIPEPS{SquareLattice, T, 5}
function SquareBCIPEPS(bulk::AT) where {AT<:AbstractArray{<:AbstractArray,2}}
    # NOTE: from here, wrapping `bulk` with a `BCIPEPS` type can prevent programing from illegal input with incorrect size.
    bulk11 = bulk[1,1]
    size(bulk11,1) == size(bulk11,2) == size(bulk11,3) == size(bulk11,4) || throw(DimensionMismatch(
        "size of tensor error, should be `(d, d, d, d, s)`, got $(size(bulk11))."))
    T = eltype(bulk11)
    BCIPEPS{SquareLattice,T,5,AT}(bulk)
end
getd(bcipeps::SquareBCIPEPS) = size(bcipeps.bulk[1,1], 1)
gets(bcipeps::SquareBCIPEPS) = size(bcipeps.bulk[1,1], 5)

"""
    indexperm_symmetrize(bcipeps::SquareBCIPEPS)

return a `SquareBCIPEPS` based on `bcipeps` that is symmetric under
permutation of its virtual indices, which is ordered below:
```
        4
        │
 1 ── bcipeps ── 3
        │
        2
```
"""
function indexperm_symmetrize(bcipeps::SquareBCIPEPS)
    Ni,Nj = size(bcipeps.bulk)
    bulk = reshape([symmetrize(bcipeps.bulk[i]) for i = 1:Ni*Nj], (Ni, Nj))
    return SquareBCIPEPS(bulk)
end

function symmetrize(x)
    x += permutedims(x, (1,4,3,2,5)) # up-down
    x += permutedims(x, (3,2,1,4,5)) # left-right
    x += permutedims(x, (2,1,4,3,5)) # diagonal
    x += permutedims(x, (4,3,2,1,5)) # rotation
    x / norm(x)
end