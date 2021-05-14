using ADBCVUMPS
using ADBCVUMPS:num_grad
using BCVUMPS:model_tensor,qrpos,lqpos,Ising,Ising22
using BCVUMPS:leftorth,leftenv,rightorth,rightenv,FLmap,ACenv,Cenv,ACCtoALAR
using Test
using Zygote
using ChainRulesTestUtils
using ChainRulesCore
using Random
using OMEinsum
using LinearAlgebra
using Random

@testset "matrix autodiff" begin
    a = randn(10, 10)
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    function foo1(x) 
        sum(Float64[x 2x; 3x x])
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

    # example to solve differential of array of array
    # use `[]` list then reshape
    A = Array{Array,2}(undef, 2, 2)
    for j = 1:2,i = 1:2
        A[i,j] = rand(2,2)
    end
    function foo2(x)
        # B[i,j] = A[i,j].*x   # mistake
        B = reshape([A[i].*x for i=1:4],2,2)
        return sum(sum(B))
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "$(Ni)x$(Nj) model_tensor" for Ni in [1,2,3], Nj in [1,2,3]
    Random.seed!(100)
    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β)
        return norm(norm(M))
    end
    @test isapprox(Zygote.gradient(foo1, 0.4)[1], num_grad(foo1, 0.4), atol=1e-8)

    function foo2(β)
        M = model_tensor(Ising22(Ni*Nj*0.1), β)
        return norm(norm(M))
    end
    @test isapprox(Zygote.gradient(foo2, 1)[1], num_grad(foo2, 1), atol=1e-8)
end

@testset "QR factorization" begin
    M = rand(10,10)
    function foo5(x)
        A = M.*x
        Q, R = qrpos(A)
        return norm(Q) + norm(R)
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1], num_grad(foo5, 1), atol = 1e-5)
end

@testset "LQ factorization" begin
    M = rand(10,10)
    function foo6(x)
        A = M .*x
        L, Q = lqpos(A)
        return norm(L) + norm(Q)
    end
    @test isapprox(Zygote.gradient(foo6, 1)[1], num_grad(foo6, 1), atol = 1e-5)
end

@testset "$(Ni)x$(Nj) leftenv and rightenv" for Ni in [1,2,3], Nj in [1,2,3]
    Random.seed!(50)
    D, d = 3, 2
    A = Array{Array,2}(undef, Ni, Nj)
    S = Array{Array,2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = rand(D, d, D)
        S[i,j] = rand(D, d, D, D, d, D)
    end

    AL, = leftorth(A) 
    _, AR, = rightorth(A)

    function foo3(β)
        M = model_tensor(Ising(Ni, Nj), β)
        AL, = leftorth(A) 
        _, AR, = rightorth(A)
        λL, FL = leftenv(AL, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(FL[i,j], S[i,j], FL[i,j])[] / ein"γcη,ηcγ -> "(FL[i,j], FL[i,j])[]
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo3, 1)[1], num_grad(foo3, 1), atol=1e-8)

    function foo4(β)
        M = model_tensor(Ising(Ni, Nj), β)
        λR, FR = rightenv(AR, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(FR[i,j], S[i,j], FR[i,j])[] / ein"γcη,ηcγ -> "(FR[i,j], FR[i,j])[]
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo4, 1)[1], num_grad(foo4, 1), atol=1e-8)
end

@testset "$(Ni)x$(Nj) ACenv and Cenv" for Ni in [1,2,3], Nj in [1,2,3]
    Random.seed!(50)
    D, d = 3, 2
    A = Array{Array,2}(undef, Ni, Nj)
    S1 = Array{Array,2}(undef, Ni, Nj)
    S2 = Array{Array,2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = rand(D, d, D)
        S1[i,j] = rand(D, d, D, D, d, D)
        S2[i,j] = rand(D, D, D, D)
    end

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    M = model_tensor(Ising(Ni, Nj), 1.0)
    _, FL = leftenv(AL, M)
    _, FR = rightenv(AR, M)

    C = Array{Array,2}(undef, Ni, Nj)
    AC = Array{Array,2}(undef, Ni, Nj)
    for j in 1:Nj,i in 1:Ni
        jr = j + 1 - (j + 1 > Nj) * Nj
        C[i,j] = L[i,j] * R[i,jr]
        AC[i,j] = ein"asc,cb -> asb"(AL[i,j], C[i,j])
    end

    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β)
        λAC, AC = ACenv(AC, FL, M, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(AC[i,j], S1[i,j], AC[i,j])[] / ein"γcη,γcη -> "(AC[i,j], AC[i,j])[]
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo1, 1)[1], num_grad(foo1, 1), atol=1e-8)

    function foo2(β)
        M = model_tensor(Ising(Ni, Nj), β)
        λL, FL = leftenv(AL, M)
        λR, FR = rightenv(AR, M)
        λC, C = Cenv(C, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γη,ηγαβ,βα -> "(C[i,j],S2[i,j],C[i,j])[]/ein"γη,γη -> "(C[i,j],C[i,j])[]
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo2, 1)[1], num_grad(foo2, 1), atol=1e-8)
end

@testset "$(Ni)x$(Nj) ACCtoALAR" for Ni in [1,2,3], Nj in [1,2,3]
    Random.seed!(10)
    D, d = 3, 2
    A = Array{Array,2}(undef, Ni, Nj)
    S1 = Array{Array,2}(undef, Ni, Nj)
    S2 = Array{Array,2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = rand(D, d, D)
        S1[i,j] = rand(D, d, D, D, d, D)
        S2[i,j] = rand(D, D, D, D)
    end

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    M = model_tensor(Ising(Ni, Nj), 1.0)
    _, FL = leftenv(AL, M)
    _, FR = rightenv(AR, M)

    C = Array{Array,2}(undef, Ni, Nj)
    for j in 1:Nj,i in 1:Ni
        jr = j + 1 - (j + 1 > Nj) * Nj
        C[i,j] = L[i,j] * R[i,jr]
    end

    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β)
        AL, C, AR = ACCtoALAR(AL, C, AR, M, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(AL[i,j], S1[i,j], AL[i,j])[] / ein"γcη,γcη -> "(AL[i,j], AL[i,j])[]
            s += ein"γη,ηγαβ,βα -> "(C[i,j],S2[i,j],C[i,j])[]/ein"γη,γη -> "(C[i,j],C[i,j])[]
            s += ein"γcη,ηcγαaβ,βaα -> "(AR[i,j], S1[i,j], AR[i,j])[] / ein"γcη,γcη -> "(AR[i,j], AR[i,j])[]
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo1, 1)[1], num_grad(foo1, 1), atol=1e-2)
end