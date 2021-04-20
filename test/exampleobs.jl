using Test
using ADBCVUMPS
using ADBCVUMPS:num_grad
using BCVUMPS
using BCVUMPS:bcvumps_env,magnetisation,magofβ,energy,eneofβ,Z,Zofβ
using Random
using Zygote
@Zygote.nograd BCVUMPS._initializect_square

@testset "$(Ni)x$(Nj) ising" for Ni = [2], Nj = [2]
    Random.seed!(100)
    model = Ising(Ni,Nj)
    β,D = 0.6,3
    function foo1(β) 
        -log(Z(bcvumps_env(model,β,D)))
    end
    @show num_grad(foo1,β)
    @test isapprox(Zygote.gradient(foo1,β)[1], num_grad(foo1,β), atol = 1e-6)
end