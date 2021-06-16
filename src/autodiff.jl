using BCVUMPS
using BCVUMPS:qrpos,lqpos,leftenv,rightenv,FLmap,FRmap,ACenv,Cenv,ACmap,Cmap,ALCtoAC,obs2x2FL,obs2x2FR,bigleftenv,bigrightenv,BgFLmap,BgFRmap
using ChainRulesCore
using FileIO
using JLD2
using KrylovKit
using LinearAlgebra
using Zygote

Zygote.@nograd BCVUMPS.StopFunction
Zygote.@nograd BCVUMPS.error
Zygote.@nograd BCVUMPS.FLint
Zygote.@nograd BCVUMPS.FRint
Zygote.@nograd BCVUMPS.BgFLint
Zygote.@nograd BCVUMPS.BgFRint
Zygote.@nograd BCVUMPS.leftorth
Zygote.@nograd BCVUMPS.rightorth
Zygote.@nograd BCVUMPS.ALCtoAC
Zygote.@nograd BCVUMPS.LRtoC
Zygote.@nograd BCVUMPS.initialA
Zygote.@nograd save
Zygote.@nograd load

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

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        dA =  As' \ dAs ./2 
        return NO_FIELDS, dA
    end
    return As, back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        M = Array(R * dR' - dQ' * Q)
        dA = (UpperTriangular(R + I * 1e-12) \ (dQ + Q * _arraytype(Q)(Symmetric(M, :L)))' )'
        return NO_FIELDS, dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        M = Array(L' * dL - dQ * Q')
        dA = LowerTriangular(L + I * 1e-12)' \ (dQ + _arraytype(Q)(Symmetric(M, :L)) * Q)
        return NO_FIELDS, dA
    end
    return (L, Q), back
end

"""
    dAMmap(Ai, Aip, Mi, L, R, j, J)

Aip means Aᵢ₊₁
```
               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──   ...  ──┐ 
               │     │              │                │ 
dMᵢJ    =     Lᵢⱼ ─ Mᵢⱼ  ── ... ──     ────── ...  ──Rᵢⱼ
               │     │              │                │ 
               └──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁J  ──  ...  ──┘ 

               ┌──  Aᵢⱼ  ── ... ──     ────── ...  ──┐ 
               │     │              │                │ 
dAᵢJ    =     Lᵢⱼ ─ Mᵢⱼ  ── ... ── dMᵢJ  ──── ...  ──Rᵢⱼ
               │     │              │                │ 
               └──  Aᵢ₊₁ⱼ  ─... ── Aᵢ₊₁J  ─── ...  ──┘ 

               ┌──  Aᵢⱼ  ── ... ── AᵢJ  ────  ...  ──┐ 
               │     │              │                │ 
Aᵢ₊₁J   =     Lᵢⱼ ─ Mᵢⱼ  ── ... ── dMᵢJ ────  ...  ──Rᵢⱼ
               │     │              │                │ 
               └──  Aᵢ₊₁ⱼ  ─... ──     ─────  ...  ──┘ 

```
"""
function dAMmap(Ai, Aip, Mi, L, R, j, J)
    Nj = size(Ai, 1)
        NL = (J - j + (J - j < 0) * Nj)
    NR = Nj - NL - 1
    L = copy(L)
    R = copy(R)
    for jj = 1:NL
        jr = j + jj - 1 - (j + jj - 1 > Nj) * Nj
        L = ein"((abc,cde),bfhd),afg -> ghe"(L, Ai[jr], Mi[jr], conj(Aip[jr]))
    end
    for jj = 1:NR
        jr = j - jj + (j - jj < 1) * Nj
        R = ein"((abc,eda),hfbd),gfc -> ehg"(R, Ai[jr], Mi[jr], conj(Aip[jr]))
    end
    dAiJ = -ein"γcη,(csap,(γsα,βaα)) -> ηpβ"(L, Mi[J], conj(Aip[J]), R)
    dAipJ = -ein"γcη,(csap,(ηpβ,βaα)) -> γsα"(L, Mi[J], Ai[J], R)
    dMiJ = -ein"(γcη,ηpβ),(γsα,βaα) -> csap"(L, Ai[J], conj(Aip[J]), R)
    return dAiJ, dAipJ, dMiJ
end

function ChainRulesCore.rrule(::typeof(leftenv), AL, M, FL; kwargs...)
    λL, FL = leftenv(AL, M, FL)
    Ni, Nj = size(AL)
    T = eltype(AL[1,1])
    atype = _arraytype(M[1,1])
    function back((dλL, dFL))
        dAL = fill!(similar(AL, atype), atype(zeros(T,size(AL[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dFL[i,j] !== nothing
                ir = i + 1 - Ni * (i == Ni)
                jr = j - 1 + Nj * (j == 1)
                ξl, info = linsolve(FR -> FRmap(AL[i,:], AL[ir,:], M[i,:], FR, jr), permutedims(dFL[i,j], (3, 2, 1)), -λL[i,j], 1)
                # errL = ein"abc,cba ->"(FL[i,j], ξl)[]
                # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. $(errL) $(info)")
                # @show info ein"abc,cba ->"(FL[i,j], ξl)[] ein"abc,abc -> "(FL[i,j], dFL[i,j])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ = dAMmap(AL[i,:], AL[ir,:], M[i,:], FL[i,j], ξl, j, J)
                    dAL[i,J] += dAiJ
                    dAL[ir,J] += dAipJ
                    dM[i,J] += dMiJ
                end
            end
        end
        return NO_FIELDS, dAL, dM, NO_FIELDS...
    end
    return (λL, FL), back
end

function ChainRulesCore.rrule(::typeof(rightenv), AR, M, FR; kwargs...)
    λR, FR = rightenv(AR, M, FR)
    Ni, Nj = size(AR)
    T = eltype(AR[1,1])
    atype = _arraytype(M[1,1])
    function back((dλ, dFR))
        dAR = fill!(similar(AR, atype), atype(zeros(T,size(AR[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = i + 1 - Ni * (i == Ni)
            jr = j - 1 + Nj * (j == 1)
            if dFR[i,jr] !== nothing
                ξr, info = linsolve(FL -> FLmap(AR[i,:], AR[ir,:], M[i,:], FL, j), permutedims(dFR[i,jr], (3, 2, 1)), -λR[i,jr], 1)
                # errR = ein"abc,cba ->"(ξr, FR[i,jr])[]
                # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. $(errR) $(info)")
                # @show info ein"abc,cba ->"(ξr, FR[i,jr])[] ein"abc,abc -> "(FR[i,jr], dFR[i,jr])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ = dAMmap(AR[i,:], AR[ir,:], M[i,:], ξr, FR[i,jr], j, J)
                    dAR[i,J] += dAiJ
                    dAR[ir,J] += dAipJ
                    dM[i,J] += dMiJ
                end
            end
        end
        return NO_FIELDS, dAR, dM, NO_FIELDS...
    end
    return (λR, FR), back
end

"""
    ACdmap(ACij, FLj, FRj, Mj, II)

```
.        .         .
.        .         .
.        .         .
│        │         │          
FLᵢ₋₁ⱼ ─ Mᵢ₋₁ⱼ ──  FRᵢ₋₁ⱼ
│        │         │   
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ
│        │         │    
└─────── ACᵢⱼ ─────┘
```
"""
function ACdmap(ACij, FLj, FRj, Mj, II)
    Ni = size(FLj,1)
    ACdm = copy(ACij)
    for i=1:Ni
        ir = II-(i-1) + (II-(i-1) < 1)*Ni
        ACdm = ein"((αaγ,αsβ),asbp),ηbβ -> γpη"(FLj[ir],ACdm,Mj[ir],FRj[ir])
    end
    return ACdm
end

"""
    ACdFMmap(FLj, Mi, FRj, AC, ACd, i, II)

```
               ┌─────  ACᵢⱼ ─────┐ 
               │        │        │ 
              FLᵢⱼ ─── Mᵢⱼ ──── FRᵢⱼ
               │        │        │ 
               ⋮         ⋮        ⋮
               │        │        │
dMIⱼ    =     FLIⱼ ───     ──── FRIⱼ 
               │        │        │
               ⋮         ⋮        ⋮
               │        │        │             
               └─────  ACdᵢ₋₁ⱼ ──┘ 

               ┌─────  ACᵢⱼ ─────┐ 
               │        │        │ 
              FLᵢⱼ ─── Mᵢⱼ  ─── FRᵢⱼ
               │        │        │ 
               ⋮         ⋮        ⋮
               │        │        │
dFLIⱼ   =        ───── MIⱼ ───  FRIⱼ 
               │        │        │
               ⋮         ⋮        ⋮
               │        │        │             
               └─────  ACdᵢ₋₁ⱼ ──┘

               ┌─────  ACᵢⱼ ─────┐ 
               │        │        │ 
              FLᵢⱼ ─── Mᵢⱼ  ─── FRᵢⱼ
               │        │        │ 
               ⋮         ⋮        ⋮
               │        │        │
dFRIⱼ   =     FLIⱼ ─── MIⱼ ─────
               │        │        │
               ⋮         ⋮        ⋮
               │        │        │             
               └─────  ACdᵢ₋₁ⱼ ──┘
```
"""
function ACdFMmap(FLj, Mj, FRj, AC, ACd, i, II)
    Ni = size(FLj, 1)
    Nu = (II - i + (II - i < 0) * Ni)
    Nd = Ni - Nu - 1
    AC = copy(AC)
    ACd = copy(ACd)
    for ii = 1:Nu
        ir = i + ii - 1 - (i + ii - 1 > Ni) * Ni
        AC = ein"((abc,cde),bhfd),efg -> ahg"(FLj[ir], AC, Mj[ir], FRj[ir])
    end
    for ii = 1:Nd
        ir = i - ii + (i - ii < 1) * Ni
        ACd = ein"((αaγ,αsβ),asbp),ηbβ -> γpη"(FLj[ir], ACd, Mj[ir], FRj[ir])
    end
    dFLIj = -ein"((ηpβ,βaα),csap),γsα -> γcη"(AC, FRj[II], Mj[II], ACd)
    dMIj = -ein"(γcη,ηpβ),(γsα,βaα) -> csap"(FLj[II], AC, ACd, FRj[II])
    dFRIj = -ein"((ηpβ,γcη),csap),γsα -> βaα"(AC, FLj[II], Mj[II], ACd)
    return dFLIj, dMIj, dFRIj
end

function ChainRulesCore.rrule(::typeof(ACenv), AC, FL, M, FR; kwargs...)
    λAC, AC = ACenv(AC, FL, M, FR)
    Ni, Nj = size(AC)
    T = eltype(AC[1,1])
    atype = _arraytype(M[1,1])
    function back((dλ, dAC))
        dFL = fill!(similar(FL, atype), atype(zeros(T,size(FL[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dFR = fill!(similar(FR, atype), atype(zeros(T,size(FR[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dAC[i,j] !== nothing
                ir = i - 1 + Ni * (i == 1)
                ξAC, info = linsolve(ACd -> ACdmap(ACd, FL[:,j], FR[:,j], M[:,j], ir), dAC[i,j], -λAC[i,j], 1)
                # errAC = ein"abc,abc ->"(AC[i,j], ξAC)[]
                # abs(errAC) > 1e-1 && throw("AC and ξ aren't orthometric. $(errAC) $(info)")
                # @show info ein"abc,abc ->"(AC[i,j], ξAC)[] ein"abc,abc -> "(AC[i,j], dAC[i,j])[]
                for II = 1:Ni
                    dFLIj, dMIj, dFRIj = ACdFMmap(FL[:,j], M[:,j], FR[:,j], AC[i,j], ξAC, i, II)
                    dFL[II,j] += dFLIj
                    dM[II,j] += dMIj
                    dFR[II,j] += dFRIj
                end
            end
        end
        return NO_FIELDS, NO_FIELDS, dFL, dM, dFR
    end
    return (λAC, AC), back
end

"""
    Cdmap(Cij, FLj, FRj, II)

```
.                .
.                .
.                .
│                │          
FLᵢ₋₁ⱼ₊₁ ─────  FRᵢ₋₁ⱼ
│                │   
FLᵢⱼ₊₁ ───────  FRᵢⱼ
│                │    
└────── Cᵢⱼ ─────┘
```
"""
function Cdmap(Cij, FLjp, FRj, II)
    Ni = size(FLjp,1)
    Cdm = copy(Cij)
    for i=1:Ni
        ir = II-(i-1) + (II-(i-1) < 1)*Ni
        Cdm = ein"(αaγ,αβ),ηaβ -> γη"(FLjp[ir],Cdm,FRj[ir])
    end
    return Cdm
end

"""
    CdFMmap(FLj, FRj, C, Cd, i, II)

```
               ┌────  Cᵢⱼ ────┐ 
               │              │ 
              FLᵢⱼ₊₁───────  FRᵢⱼ
               │              │ 
               ⋮               ⋮
               │              │
dFLIⱼ₊₁ =        ──────────  FRIⱼ
               │              │
               ⋮               ⋮
               │              │             
               └──── Cdᵢⱼ ────┘ 

               ┌────  Cᵢⱼ ────┐ 
               │              │ 
              FLᵢⱼ₊₁ ──────  FRᵢⱼ
               │              │ 
               ⋮               ⋮
               │              │
dFRIⱼ   =     FLᵢ₊Iⱼ₊₁ ──────  
               │              │
               ⋮               ⋮
               │              │               
               └──── Cdᵢⱼ ────┘ 
```
"""
function CdFMmap(FLjp, FRj, C, Cd, i, II)
    Ni = size(FLjp, 1)
    Nu = (II - i + (II - i < 0) * Ni)
    Nd = Ni - Nu - 1
    C = copy(C)
    Cd = copy(Cd)
    for ii = 1:Nu
        ir = i + ii - 1 - (i + ii - 1 > Ni) * Ni
        C = ein"(αaγ,γη),ηaβ -> αβ"(FLjp[ir], C, FRj[ir])
    end
    for ii = 1:Nd
        ir = i - ii + (i - ii < 1) * Ni
        Cd = ein"(αaγ,αβ),ηaβ -> γη"(FLjp[ir], Cd, FRj[ir])
    end
    dFLIjp = -ein"(ηβ,βaα),γα -> γaη"(C, FRj[II], Cd)
    dFRIj = -ein"(ηβ,γcη),γα -> βcα"(C, FLjp[II], Cd)
    return dFLIjp, dFRIj
end

function ChainRulesCore.rrule(::typeof(Cenv), C, FL, FR; kwargs...)
    λC, C = Cenv(C, FL, FR)
    Ni, Nj = size(C)
    T = eltype(C[1,1])
    atype = _arraytype(FL[1,1])
    function back((dλ, dC))
        dFL = fill!(similar(FL, atype), atype(zeros(T,size(FL[1,1]))))
        dFR = fill!(similar(FR, atype), atype(zeros(T,size(FR[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dC[i,j] !== nothing
                ir = i - 1 + Ni * (i == 1)
                jr = j + 1 - (j==Nj) * Nj
                ξC, info = linsolve(Cd -> Cdmap(Cd, FL[:,jr], FR[:,j], ir), dC[i,j], -λC[i,j], 1)
                # errC = ein"ab,ab ->"(C[i,j], ξC)[]
                # abs(errC) > 1e-1 && throw("C and ξ aren't orthometric. $(errC) $(info)")
                # @show info ein"ab,ab ->"(C[i,j], ξC)[] ein"ab,ab -> "(C[i,j], dC[i,j])[]
                for II = 1:Ni
                    dFLIjp, dFRIj = CdFMmap(FL[:,jr], FR[:,j], C[i,j], ξC, i, II)
                    dFL[II,jr] += dFLIjp
                    dFR[II,j] += dFRIj
                end
            end
        end
        return NO_FIELDS, NO_FIELDS, dFL, dFR
    end
    return (λC, C), back
end

function ChainRulesCore.rrule(::typeof(obs2x2FL), AL, AR, M, FL; kwargs...)
    λL, FL = obs2x2FL(AL, AR, M, FL)
    Ni, Nj = size(AL)
    T = eltype(AL[1,1])
    atype = _arraytype(M[1,1])
    ALd = reshape([permutedims(AR[i], (3, 2, 1)) for i = 1:4], (2,2))
    function back((dλL, dFL))
        dAL = fill!(similar(AL, atype), atype(zeros(T,size(AL[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dAR = fill!(similar(AR, atype), atype(zeros(T,size(AR[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dFL[i,j] !== nothing
                ir = Ni + 1 - i
                jr = j - 1 + Nj * (j == 1)
                ξl, info = linsolve(FR -> FRmap(AL[i,:], ALd[ir,:], M[i,:], FR, jr), permutedims(dFL[i,j], (3, 2, 1)), -λL[i,j], 1)
                # errL = ein"abc,cba ->"(FL[i,j], ξl)[]
                # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. $(errL) $(info)")
                # @show info ein"abc,cba ->"(FL[i,j], ξl)[] ein"abc,abc -> "(FL[i,j], dFL[i,j])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ = dAMmap(AL[i,:], ALd[ir,:], M[i,:], FL[i,j], ξl, j, J)
                    dAL[i,J] += dAiJ
                    dAR[ir,J] += permutedims(dAipJ, (3, 2, 1))
                    dM[i,J] += dMiJ
                end
            end
        end
        return NO_FIELDS, dAL, dAR, dM, NO_FIELDS...
    end
    return (λL, FL), back
end

function ChainRulesCore.rrule(::typeof(obs2x2FR), AR, AL, M, FR; kwargs...)
    λR, FR = obs2x2FR(AR, AL, M, FR)
    Ni, Nj = size(AR)
    T = eltype(AR[1,1])
    atype = _arraytype(M[1,1])
    ARd = reshape([permutedims(AL[i], (3, 2, 1)) for i = 1:4], (2,2))
    function back((dλ, dFR))
        dAR = fill!(similar(AR, atype), atype(zeros(T,size(AR[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dAL = fill!(similar(AL, atype), atype(zeros(T,size(AL[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j - 1 + Nj * (j == 1)
            if dFR[i,jr] !== nothing
                ξr, info = linsolve(FL -> FLmap(AR[i,:], ARd[ir,:], M[i,:], FL, j), permutedims(dFR[i,jr], (3, 2, 1)), -λR[i,jr], 1)
                # errR = ein"abc,cba ->"(ξr, FR[i,jr])[]
                # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. $(errR) $(info)")
                # @show info ein"abc,cba ->"(ξr, FR[i,jr])[] ein"abc,abc -> "(FR[i,jr], dFR[i,jr])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ = dAMmap(AR[i,:], ARd[ir,:], M[i,:], ξr, FR[i,jr], j, J)
                    dAR[i,J] += dAiJ
                    dAL[ir,J] += permutedims(dAipJ, (3, 2, 1))
                    dM[i,J] += dMiJ
                end
            end
        end
        return NO_FIELDS, dAR, dAL, dM, NO_FIELDS...
    end
    return (λR, FR), back
end

"""
    dBgAMmap(Ai, Aip, Mi, Mip, L, R, j, J)

Aip means Aᵢ₊₁
```
               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ...  ──   ────── ...   ──│ 
dMᵢJ    =     Lᵢⱼ    │              │                Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ──Mᵢ₊₁J ───  ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─  ...── AᵢₚJ  ──── ...   ──┘ 

               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ...  ──MᵢJ   ─── ...   ──│ 
dMᵢₚJ    =    Lᵢⱼ    │              │               Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ──     ───   ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─  ...── AᵢₚJ  ──── ...   ──┘ 

               ┌──  Aᵢⱼ  ── ... ──       ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ... ── MᵢJ ────  ...   ──│ 
dAᵢJ    =     Lᵢⱼ    │              │                Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ── Mᵢ₊₁J ─── ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─  ...── AᵢₚJ  ──── ...   ──┘ 

               ┌──  Aᵢⱼ  ── ... ── AᵢJ   ──  ...   ──┐ 
               │     │              │                │ 
               │─   Mᵢⱼ  ── ... ── MᵢJ ────  ...   ──│ 
dAᵢₚJ    =     Lᵢⱼ    │              │               Rᵢⱼ 
               │─   Mᵢ₊₁ⱼ ──... ── Mᵢ₊₁J ─── ...   ──│ 
               │     │              │                │
               └──  Aᵢₚⱼ  ─ ... ──      ──── ...   ──┘ 

```
"""
function dBgAMmap(Ai, Aip, Mi, Mip, L, R, j, J)
    Nj = size(Ai, 1)
        NL = (J - j + (J - j < 0) * Nj)
    NR = Nj - NL - 1
    L = copy(L)
    R = copy(R)
    for jj = 1:NL
        jr = j + jj - 1 - (j + jj - 1 > Nj) * Nj
        L = ein"(((adgi,abc),dfeb),gjhf),ijk -> cehk"(L, Ai[jr], Mi[jr], Mip[jr], conj(Aip[jr]))
    end
    for jj = 1:NR
        jr = j - jj + (j - jj < 1) * Nj
        R = ein"(((cehk,abc),dfeb),gjhf),ijk -> adgi"(R, Ai[jr], Mi[jr], Mip[jr], conj(Aip[jr]))
    end
    dAiJ = -ein"(((adgi,ijk),gjhf),dfeb), cehk -> abc"(L, Aip[J], Mip[J], Mi[J], R)
    dAipJ = -ein"(((adgi,abc),dfeb),gjhf), cehk -> ijk"(L, Ai[J], Mi[J], Mip[J], R)
    dMiJ = -ein"(adgi,abc),(gjhf,(ijk,cehk)) -> dfeb"(L, Ai[J], Mip[J], Aip[J], R)
    dMipJ = -ein"((adgi,abc),dfeb),(ijk,cehk)-> gjhf"(L, Ai[J], Mi[J], Aip[J], R)
    return dAiJ, dAipJ, dMiJ, dMipJ
end

function ChainRulesCore.rrule(::typeof(bigleftenv), AL, AR, M, BgFL; kwargs...)
    λL, BgFL = bigleftenv(AL, AR, M, BgFL)
    Ni, Nj = size(AL)
    T = eltype(AL[1,1])
    atype = _arraytype(M[1,1])
    ALd = reshape([permutedims(AR[i], (3, 2, 1)) for i = 1:4], (2,2))
    function back((dλL, dBgFL))
        dAL = fill!(similar(AL, atype), atype(zeros(T,size(AL[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dAR = fill!(similar(AR, atype), atype(zeros(T,size(AR[1,1]))))
        for j = 1:Nj, i = 1:Ni
            if dBgFL[i,j] !== nothing
                ir = Ni + 1 - i
                jr = j - 1 + Nj * (j == 1)
                ξl, info = linsolve(BgFR -> BgFRmap(AL[i,:], ALd[i,:], M[i,:], M[ir,:], BgFR, jr), dBgFL[i,j], -λL[i,j], 1)
                # errL = ein"abc,cba ->"(FL[i,j], ξl)[]
                # abs(errL) > 1e-1 && throw("FL and ξl aren't orthometric. $(errL) $(info)")
                # @show info ein"abc,cba ->"(FL[i,j], ξl)[] ein"abc,abc -> "(FL[i,j], dFL[i,j])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ, dMipJ = dBgAMmap(AL[i,:], ALd[i,:], M[i,:], M[ir,:], BgFL[i,j], ξl, j, J)
                    dAL[i,J] += dAiJ
                    dAR[i,J] += permutedims(dAipJ, (3,2,1))
                    dM[i,J] += dMiJ
                    dM[ir,J] += dMipJ
                end
            end
        end
        return NO_FIELDS, dAL, dAR, dM, NO_FIELDS
    end
    return (λL, BgFL), back
end

function ChainRulesCore.rrule(::typeof(bigrightenv), AR, AL, M, BgFR; kwargs...)
    λR, BgFR = bigrightenv(AR, AL, M, BgFR)
    Ni, Nj = size(AR)
    T = eltype(AR[1,1])
    atype = _arraytype(M[1,1])
    ARd = reshape([permutedims(AL[i], (3, 2, 1)) for i = 1:4], (2,2))
    function back((dλ, dBgFR))
        dAR = fill!(similar(AR, atype), atype(zeros(T,size(AR[1,1]))))
        dM = fill!(similar(M, atype), atype(zeros(T,size(M[1,1]))))
        dAL = fill!(similar(AL, atype), atype(zeros(T,size(AL[1,1]))))
        for j = 1:Nj, i = 1:Ni
            ir = Ni + 1 - i
            jr = j - 1 + Nj * (j == 1)
            if dBgFR[i,jr] !== nothing
                ξr, info = linsolve(BgFL -> BgFLmap(AR[i,:], ARd[i,:], M[i,:], M[ir,:], BgFL, j), dBgFR[i,jr], -λR[i,jr], 1)
                # errR = ein"abc,cba ->"(ξr, FR[i,jr])[]
                # abs(errR) > 1e-1 && throw("FR and ξr aren't orthometric. $(errR) $(info)")
                # @show info ein"abc,cba ->"(ξr, FR[i,jr])[] ein"abc,abc -> "(FR[i,jr], dFR[i,jr])[]
                for J = 1:Nj
                    dAiJ, dAipJ, dMiJ, dMipJ = dBgAMmap(AR[i,:], ARd[i,:], M[i,:], M[ir,:], ξr, BgFR[i,jr], j, J)
                    dAR[i,J] += dAiJ
                    dAL[i,J] += permutedims(dAipJ, (3,2,1))
                    dM[i,J] += dMiJ
                    dM[ir,J] += dMipJ
                end
            end
        end
        return NO_FIELDS, dAR, dAL, dM, NO_FIELDS
    end
    return (λR, BgFR), back
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
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
        num_grad(foo, b[i], δ=δ)
    end
    return _arraytype(a)(df)
end