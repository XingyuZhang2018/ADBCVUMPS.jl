using Test
using ADBCVUMPS
using ADBCVUMPS:num_grad
using BCVUMPS
using BCVUMPS:bcvumps_env,magnetisation,magofβ,energy,eneofβ,Z,Zofβ
using Random
using Zygote

@testset "$(Ni)x$(Nj) ising" for Ni = [1,2,3], Nj = [1,2,3]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    β,D = 0.5, 10
    function foo1(β)
        -log(Z(bcvumps_env(model,β,D)))
    end
    function foo2(β)
        magnetisation(bcvumps_env(model,β,D; maxiter = 20, tol = 1e-12),model,β)
    end

    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-6)
    @test isapprox(Zygote.gradient(foo2,β)[1],num_grad(foo2,β), atol = 1e-6)    
end

@testset "J1-J2-2x2-ising" begin
    Random.seed!(100)
    model = Ising22(2.0)
    β,D = 0.5,10
    function foo1(β) 
        -log(Z(bcvumps_env(model,β,D)))
    end
    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-6)
end