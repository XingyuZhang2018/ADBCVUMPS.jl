using ADBCVUMPS
using ADBCVUMPS:num_grad, safetr
using BCVUMPS:model_tensor,qrpos,lqpos,Ising,Ising22
using BCVUMPS:leftorth,leftenv,rightorth,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,obs2x2FL,obs2x2FR
using ChainRulesCore
using CUDA
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

@testset "matrix autodiff with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    a = atype(randn(10, 10))
    @test Zygote.gradient(norm, a)[1] ≈ num_grad(norm, a)

    function foo1(x) 
        sum(atype(Float64[x 2x; 3x x]))
    end
    @test Zygote.gradient(foo1, 1)[1] ≈ num_grad(foo1, 1)

    # example to solve differential of array of array
    # use `[]` list then reshape
    A = Array{atype,2}(undef, 2, 2)
    for j = 1:2,i = 1:2
        A[i,j] = atype(rand(dtype,2,2))
    end
    function foo2(x)
        # B[i,j] = A[i,j].*x   # mistake
        B = reshape([A[i]*x for i=1:4],2,2)
        return sum(sum(B))
    end
    @test Zygote.gradient(foo2, 1)[1] ≈ num_grad(foo2, 1)
end

@testset "$(Ni)x$(Nj) model_tensor with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [1,2,3], Nj = [1,2,3]
    Random.seed!(100)
    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        return norm(norm(M))
    end
    @test isapprox(Zygote.gradient(foo1, 0.4)[1], num_grad(foo1, 0.4), atol=1e-8)

    function foo2(β)
        M = model_tensor(Ising22(Ni*Nj*0.1), β; atype = atype)
        return norm(norm(M))
    end
    @test isapprox(Zygote.gradient(foo2, 1)[1], num_grad(foo2, 1), atol=1e-8)
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    M = atype(rand(10,10))
    function foo5(x)
        A = M .* x
        Q, R = qrpos(A)
        return norm(Q) + norm(R)
    end
    @test isapprox(Zygote.gradient(foo5, 1)[1], num_grad(foo5, 1), atol = 1e-5)
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64]
    M = atype(rand(10,10))
    function foo6(x)
        A = M .*x
        L, Q = lqpos(A)
        return norm(L) + norm(Q)
    end
    @test isapprox(Zygote.gradient(foo6, 1)[1], num_grad(foo6, 1), atol = 1e-5)
end

@testset "loop_einsum mistake with $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    D = 10
    A = atype(rand(D,D,D))
    B = atype(rand(D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = ein"abc,abd -> cd"(C,C)
        F = ein"ab,ac -> bc"(D,D)
        return safetr(E)/safetr(F)
    end 
    @time @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "$(Ni)x$(Nj) leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S = Array{atype{dtype,6},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S[i,j] = atype(rand(dtype, D, d, D, D, d, D))
    end

    AL, = leftorth(A) 
    _, AR, = rightorth(A)

    function foo3(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        _, FL = leftenv(AL, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γcη,ηcγαaβ,daα -> βd"(FL[i,j], S[i,j], FL[i,j])
            B = ein"γcη,ηca -> γa"(FL[i,j], FL[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo3, 1)[1], num_grad(foo3, 1), atol=1e-8)

    function foo4(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        _, FR = rightenv(AR, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γcη,ηcγαaβ,daα -> βd"(FR[i,j], S[i,j], FR[i,j])
            B = ein"γcη,ηca -> γa"(FR[i,j], FR[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo4, 1)[1], num_grad(foo4, 1), atol=1e-8)
end

@testset "$(Ni)x$(Nj) ACenv and Cenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S1 = Array{atype{dtype,6},2}(undef, Ni, Nj)
    S2 = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S1[i,j] = atype(rand(dtype, D, d, D, D, d, D))
        S2[i,j] = atype(rand(dtype, D, D, D, D))
    end

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    M = model_tensor(Ising(Ni, Nj), 1.0; atype = atype)
    _, FL = leftenv(AL, M)
    _, FR = rightenv(AR, M)

    C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        _, AC = ACenv(AC, FL, M, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γcη,ηcγαaβ,daα -> βd"(AC[i,j], S1[i,j], AC[i,j])
            B = ein"γcη,γca -> ηa"(AC[i,j], AC[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo1, 1)[1], num_grad(foo1, 1), atol=1e-8)

    function foo2(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        _, FL = leftenv(AL, M)
        _, FR = rightenv(AR, M)
        _, C = Cenv(C, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γη,ηγαβ,dα -> βd"(C[i,j], S2[i,j], C[i,j])
            B = ein"γη,γd -> ηd"(C[i,j], C[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo2, 1)[1], num_grad(foo2, 1), atol=1e-8)
end

@testset "$(Ni)x$(Nj) ACCtoALAR with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S1 = Array{atype{dtype,6},2}(undef, Ni, Nj)
    S2 = Array{atype{dtype,4},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S1[i,j] = atype(rand(dtype, D, d, D, D, d, D))
        S2[i,j] = atype(rand(dtype, D, D, D, D))
    end

    AL, L, _ = leftorth(A) 
    R, AR, _ = rightorth(A)
    M = model_tensor(Ising(Ni, Nj), 1.0; atype = atype)
    _, FL = leftenv(AL, M)
    _, FR = rightenv(AR, M)

    C = LRtoC(L, R)

    function foo1(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        AL, C, AR = ACCtoALAR(AL, C, AR, M, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γcη,ηcγαaβ,daα -> βd"(AL[i,j], S1[i,j], AL[i,j])
            B = ein"γcη,γca -> ηa"(AL[i,j], AL[i,j])
            s += safetr(A)/safetr(B)
            A = ein"γcη,ηcγαaβ,daα -> βd"(AR[i,j], S1[i,j], AR[i,j])
            B = ein"γcη,γca -> ηa"(AR[i,j], AR[i,j])
            s += safetr(A)/safetr(B)
            A = ein"γη,ηγαβ,dα -> βd"(C[i,j], S2[i,j], C[i,j])
            B = ein"γη,γd -> ηd"(C[i,j], C[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end
    @test isapprox(Zygote.gradient(foo1, 1)[1], num_grad(foo1, 1), atol=1e-4)
end

@testset "obs2x2 leftenv and rightenv with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(50)
    D, d = 3, 2
    A = Array{atype{dtype,3},2}(undef, Ni, Nj)
    S = Array{atype{dtype,6},2}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = atype(rand(dtype, D, d, D))
        S[i,j] = atype(rand(dtype, D, d, D, D, d, D))
    end

    AL, = leftorth(A) 
    _, AR, = rightorth(A)

    function foo3(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        _, FL = obs2x2FL(AL, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γcη,ηcγαaβ,daα -> βd"(FL[i,j], S[i,j], FL[i,j])
            B = ein"γcη,ηca -> γa"(FL[i,j], FL[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo3, 1)[1], num_grad(foo3, 1), atol=1e-8)

    function foo4(β)
        M = model_tensor(Ising(Ni, Nj), β; atype = atype)
        _, FR = obs2x2FL(AR, M)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = ein"γcη,ηcγαaβ,daα -> βd"(FR[i,j], S[i,j], FR[i,j])
            B = ein"γcη,ηca -> γa"(FR[i,j], FR[i,j])
            s += safetr(A)/safetr(B)
        end
        return s
    end 
    @test isapprox(Zygote.gradient(foo4, 1)[1], num_grad(foo4, 1), atol=1e-8)
end