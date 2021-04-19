using ADBCVUMPS
using ADBCVUMPS:num_grad,model_tensor
using BCVUMPS:qrpos,lqpos,leftorth,leftenv!,rightorth,rightenv!,FLmap,Ising,ACenv!,Cenv!
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
        sum(Float64[x 2x; 3x 4x])
    end
    a = Float64[1 2; 3 4]

    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)
end

@testset "QR factorization" begin
    function foo5(x)
        A = [1. + 1im 2 3;2 2 3;3 3 3] .* x
        Q, R = qrpos(A)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo5, 1)[1] ≈ num_grad(foo5, 1)
    test_rrule(qrpos, rand(10, 10))
end

@testset "LQ factorization" begin
    function foo6(x)
        A = [1. + 1im 2 3;2 2 3;3 3 3] .* x
        L, Q = lqpos(A)
        return norm(L) + norm(Q)
    end
    @test Zygote.gradient(foo6, 1)[1] ≈ num_grad(foo6, 1)
    test_rrule(lqpos, rand(10, 10))
end

@testset "$(Ni)x$(Nj) model_tensor" for Ni in [2,3], Nj in [2,3]
    function foo3(β)
        M = model_tensor(Ising(Ni, Nj), β)
        return norm(norm(M))
    end
    @test isapprox(Zygote.gradient(foo3, 1)[1], num_grad(foo3, 1), atol=1e-8)
end

@testset "$(Ni)x$(Nj) leftenv and rightenv" for Ni in [2,3], Nj in [2,3]
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
        λL, FL = leftenv!(AL, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(FL[i,j], S[i,j], FL[i,j])[] / ein"γcη,ηcγ -> "(FL[i,j], FL[i,j])[]
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo3, 1)[1], num_grad(foo3, 1), atol=1e-8)

    function foo4(β)
        M = model_tensor(Ising(Ni, Nj), β)
        λR, FR = rightenv!(AR, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(FR[i,j], S[i,j], FR[i,j])[] / ein"γcη,ηcγ -> "(FR[i,j], FR[i,j])[]
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo4, 1)[1], num_grad(foo4, 1), atol=1e-8)
end

@testset "$(Ni)x$(Nj) ACenv and Cenv" for Ni in [2,3], Nj in [2,3]
    Random.seed!(100)
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
    _, FL = leftenv!(AL, M)
    _, FR = rightenv!(AR, M)

    C = Array{Array,2}(undef, Ni, Nj)
    AC = Array{Array,2}(undef, Ni, Nj)
    for j in 1:Nj,i in 1:Ni
        jr = j + 1 - (j + 1 > Nj) * Nj
        C[i,j] = L[i,j] * R[i,jr]
        AC[i,j] = ein"asc,cb -> asb"(AL[i,j], C[i,j])
    end

    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β)
        λAC, AC = ACenv!(AC, FL, M, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γcη,ηcγαaβ,βaα -> "(AC[i,j], S1[i,j], AC[i,j])[] / ein"γcη,γcη -> "(AC[i,j], AC[i,j])[]
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo1, 1)[1], num_grad(foo1, 1), atol=1e-8)

    function foo2(β)
        M = model_tensor(Ising(Ni, Nj), β)
        λL, FL = leftenv!(AL, M)
        λR, FR = rightenv!(AR, M)
        λC, C = Cenv!(C, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            s += ein"γη,ηγαβ,βα -> "(C[i,j],S2[i,j],C[i,j])[]/ein"γη,γη -> "(C[i,j],C[i,j])[]
        end
        return s
    end
    # println(num_grad(foo2, 1))
    @test isapprox(Zygote.gradient(foo2, 1)[1], num_grad(foo2, 1), atol=1e-5)
end