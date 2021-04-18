using ChainRulesCore
using LinearAlgebra
using KrylovKit
using BCVUMPS:qrpos,lqpos,leftenv!,rightenv!,FLmap,FRmap

@Zygote.nograd BCVUMPS.StopFunction
@Zygote.nograd BCVUMPS.leftorth
@Zygote.nograd BCVUMPS.rightorth
@Zygote.nograd BCVUMPS.FLint
@Zygote.nograd BCVUMPS.FRint
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
    dAMmap(Ai, Aip, Mi, L, R, j, J)

Aip means Aᵢ₊₁
```
               ┌──  Aᵢⱼ  ── ... ── Aᵢⱼ₊J   ── ...  ──┐ 
               │     │              │                │ 
dMᵢⱼ₊J   =    Lᵢⱼ ─ Mᵢⱼ  ── ... ──     ──     ...  ──Rᵢⱼ
               │     │              │                │ 
               ┕──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁ⱼ₊J  ──...  ──┘ 

               ┌──  Aᵢⱼ  ── ... ──     ────── ...  ──┐ 
               │     │              │                │ 
dAᵢⱼ₊J   =    Lᵢⱼ ─ Mᵢⱼ  ── ... ── dMᵢⱼ₊J  ── ...  ──Rᵢⱼ
               │     │              │                │ 
               ┕──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁ⱼ₊J  ──...  ──┘ 

               ┌──  Aᵢⱼ  ── ... ── Aᵢⱼ₊J  ──  ...  ──┐ 
               │     │              │                │ 
Aᵢ₊₁ⱼ₊J   =   Lᵢⱼ ─ Mᵢⱼ  ── ... ── dMᵢⱼ₊J  ── ...  ──Rᵢⱼ
               │     │              │                │ 
               ┕──  Aᵢ₊₁ⱼ  ─... ──     ───────...  ──┘ 

```
"""
function dAMmap(Ai, Aip, Mi, L, R, j, J)
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
    dAiJ = -ein"γcη,csap,γsα,βaα -> ηpβ"(L, Mi[J], conj(Aip[J]), R)
    dAipJ = -ein"γcη,csap,ηpβ,βaα -> γsα"(L, Mi[J], Ai[J], R)
    dMiJ = -ein"γcη,ηpβ,γsα,βaα -> csap"(L, Ai[J], conj(Aip[J]), R)
    return dAiJ, dAipJ, dMiJ
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
            for J = 1:Nj
                dAiJ, dAipJ, dMiJ = dAMmap(AL[i,:], AL[ir,:], M[i,:], FL[i,j], ξl, j, J)
                dAL[i,J] += dAiJ
                dAL[ir,J] += dAipJ
                dM[i,J] += dMiJ
            end
            # @show info ein"abc,cba ->"(FL[i,j], ξl)[] ein"abc,abc -> "(FL[i,j], dFL[i,j])[]
        end
        return NO_FIELDS, dAL, dM, NO_FIELDS...
    end
    return (λL, FL), back
end

function ChainRulesCore.rrule(::typeof(rightenv!), AR, M, FR; kwargs...)
    λR, FR = rightenv!(AR, M, FR; kwargs...)
    Ni, Nj = size(AR)
    T = eltype(AR[1,1])
    function back((dλ, dFR))
        dAR = fill!(similar(AR, Array), zeros(size(AR[1,1])))
        dM = fill!(similar(M, Array), zeros(size(M[1,1])))
        for j in 1:Nj, i in 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            ξr = rand(T, size(FR[i,j]))
            ξr = FLmap(AR[i,:], AR[ir,:], M[i,:], ξr, j) - λR[i,jr] .* ξr
            ξr, info = linsolve(FL -> FLmap(AR[i,:], AR[ir,:], M[i,:], FL, j), permutedims(dFR[i,jr], (3, 2, 1)), ξr, -λR[i,jr], 1; kwargs...)
            for J = 1:Nj
                dAiJ, dAipJ, dMiJ = dAMmap(AR[i,:], AR[ir,:], M[i,:], ξr, FR[i,jr], j, J)
                dAR[i,J] += dAiJ
                dAR[ir,J] += dAipJ
                dM[i,J] += dMiJ
            end
            # @show info ein"abc,cba ->"(FR[i,jr], ξr)[] ein"abc,abc -> "(FR[i,jr], dFR[i,jr])[]
        end
        return NO_FIELDS, dAR, dM, NO_FIELDS...
    end
    return (λR, FR), back
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