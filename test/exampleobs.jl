using ADBCVUMPS
using ADBCVUMPS:num_grad
using BCVUMPS
using BCVUMPS:obs_bcenv,magnetisation,Z
using Random
using Test
using Zygote

@testset "$(Ni)x$(Nj) ising with $atype{$dtype}" for atype in [Array], dtype in [Float64], Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    β,χ = 0.6, 10
    # function foo1(β)
    #     M = model_tensor(model, β; atype = atype)
    #     env = obs_bcenv(model, M; atype = atype, D = 2, χ = 10, tol = 1e-10, maxiter = 10, verbose = true)
    #     -log(Z(env))
    # end
    function foo2(β)
        M = model_tensor(model, β; atype = atype)
        env = obs_bcenv(model, M; atype = atype, D = 2, χ = χ, tol = 1e-10, maxiter = 10, verbose = true)
        magnetisation(env,model,β)
    end

    @show foo2(β)
    # @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-6)
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