using Test
using ADBCVUMPS
using ADBCVUMPS:num_grad
using BCVUMPS
using BCVUMPS:bcvumps_env,magnetisation,Z
using Random
using Zygote

@testset "$(Ni)x$(Nj) ising with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    β,D = 0.5, 10
    function foo1(β)
        -log(Z(bcvumps_env(model,β,D; atype = atype)))
    end
    function foo2(β)
        magnetisation(bcvumps_env(model,β,D; maxiter = 20, tol = 1e-12, atype = atype),model,β)
    end

    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-6)
    @test isapprox(Zygote.gradient(foo2,β)[1], num_grad(foo2,β), atol = 1e-6)    
end

@testset "J1-J2-2x2-ising with $atype{$dtype}" for atype in [Array, CuArray], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising22(2.0)
    β,D = 0.5,10
    function foo1(β) 
        -log(Z(bcvumps_env(model,β,D; atype = atype)))
    end
    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-6)
end