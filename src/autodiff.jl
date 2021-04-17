using ChainRulesCore
using LinearAlgebra
using KrylovKit
using BCVUMPS:qrpos,lqpos,leftenv!,FLmap,FRmap

@Zygote.nograd BCVUMPS.StopFunction
@Zygote.nograd BCVUMPS.leftorth
@Zygote.nograd BCVUMPS.rightorth
@Zygote.nograd BCVUMPS.FLint
@Zygote.nograd BCVUMPS._initializect_square

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NO_FIELDS, NO_FIELDS, NO_FIELDS, permutedims(ȳ)...
    end
    return y, back
end

# improves performance compared to default implementation, also avoids errors
# with some complex arrays
function ChainRulesCore.rrule(::typeof(LinearAlgebra.norm), A::AbstractArray)
    n = norm(A)
    function back(Δ)
        return NO_FIELDS, Δ .* A ./ (n + eps(0f0)), NO_FIELDS
    end
    return n, back
end

function ChainRulesCore.rrule(::typeof(CopyM), M, Ni, Nj)
    function back(dm)
        dM = zeros(size(dm[1,1]))
        for j in 1:Nj, i in 1:Ni
            dM += dm[i,j]
        end
        return NO_FIELDS, dM, NO_FIELDS, NO_FIELDS
    end
    return CopyM(M, Ni, Nj), back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = R * dR' - dQ' * Q
        Rt = rand(T, size(R))
        Rt, info = linsolve(x -> R * x, Matrix(I, size(R)), Rt, 0, 1)
        dA = (dQ + Q * Symmetric(M, :L)) * (Rt)'
        return NO_FIELDS, dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = L' * dL - dQ * Q'
        Lt = rand(T, size(L))
        Lt, info = linsolve(x -> L * x, Matrix(I, size(L)), Lt, 0, 1)
        dA = (Lt)' * (dQ + Symmetric(M, :L) * Q)
        return NO_FIELDS, dA
    end
    return (L, Q), back
end

"""
    dMmap(Ai, Aip, Mi, L, R, j, J)

Aip means Aᵢ₊₁
```
               ┌──  Aᵢⱼ  ── ... ── Aᵢⱼ₊J   ── ...  ──┐ 
               │     │              │                │ 
dMᵢⱼ₊J   =    Lᵢⱼ ─ Mᵢⱼ  ── ... ──     ──     ...  ──Rᵢⱼ
               │     │              │                │ 
               ┕──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁ⱼ₊J  ──...  ──┘ 

```
"""
function dMmap(Ai, Aip, Mi, L, R, j, J)
    Nj = size(Ai, 1)
        NL = (J - j + (J - j < 0) * Nj)
    NR = Nj - NL - 1
    for jj = 1:NL
        jr = j + jj - 1 - (j + jj - 1 > Nj) * Nj
        L = ein"abc,cde,bfhd,afg -> ghe"(L, Ai[jr], Mi[jr], conj(Aip[jr]))
    end
    for jj = 1:NR
        jr = j + Nj - jj - (j + Nj - jj > Nj) * Nj
        R = ein"abc,eda,hfbd,gfc -> ehg"(R, Ai[jr], Mi[jr], conj(Aip[jr]))
    end
    dM = -ein"γcη,ηpβ,γsα,βaα -> csap"(L, Ai[J], conj(Aip[J]), R)
    return dM
end

function ChainRulesCore.rrule(::typeof(leftenv!), AL, M, FL; kwargs...)
    λL, FL = leftenv!(AL, M, FL; kwargs...)
    Ni, Nj = size(AL)
    T = eltype(AL[1,1])
    function back((dλ, dFL))
        dAL = fill!(similar(AL, Array), zeros(size(AL[1,1])))
        dM = fill!(similar(M, Array), zeros(size(M[1,1])))
        for j in 1:Nj, i in 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            ξl = rand(T, size(FL[i,j]))
            ξl = FRmap(AL[i,:], AL[ir,:], M[i,:], ξl, jr) - λL[i,j] .* ξl
            ξl, info = linsolve(FR -> FRmap(AL[i,:], AL[ir,:], M[i,:], FR, jr), permutedims(dFL[i,j], (3, 2, 1)), ξl, -λL[i,j], 1; kwargs...)
            #####     To do: correct dAL      #####
            dAL[i,j] += - ein"γcη,csap,γsα,βaα -> ηpβ"(FL[i,j], M[i,j], conj(AL[ir,j]), ξl) .+ 1
            dAL[ir,j] += - ein"γcη,csap,ηpβ,βaα -> γsα"(FL[i,j], M[i,j], AL[i,j], ξl) .+ 99
            for J = 1:Nj
                dM[i,J] += dMmap(AL[i,:], AL[ir,:], M[i,:], FL[i,j], ξl, j, J)
            end
            # @show info ein"abc,cba ->"(FL[i,j], ξl)[] ein"abc,abc -> "(FL[i,j], dFL[i,j])[]
        end
        return NO_FIELDS, dAL, dM, NO_FIELDS...
    end
    return (λL, FL), back
end

@doc raw"
    num_grad(f, K::Real; [δ = 1e-5])

return the numerical gradient of `f` at `K` calculated with
`(f(K+δ/2) - f(K-δ/2))/δ`

# example

```jldoctest; setup = :(using TensorNetworkAD)
julia> TensorNetworkAD.num_grad(x -> x * x, 3) ≈ 6
true
```
"
num_grad(f, K::Real; δ::Real=1e-5) = (f(K + δ / 2) - f(K - δ / 2)) / δ

@doc raw"
    num_grad(f, K::AbstractArray; [δ = 1e-5])
    
return the numerical gradient of `f` for each element of `K`.

# example

```jldoctest; setup = :(using TensorNetworkAD, LinearAlgebra)
julia> TensorNetworkAD.num_grad(tr, rand(2,2)) ≈ I
true
```
"
function num_grad(f, a::AbstractArray; δ::Real=1e-5)
map(CartesianIndices(a)) do i
        foo = x -> (ac = copy(a); ac[i] = x; f(ac))
        num_grad(foo, a[i], δ=δ)
    end
end