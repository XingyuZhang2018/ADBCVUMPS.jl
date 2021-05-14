using Test
using BCVUMPS
using ADBCVUMPS
using ADBCVUMPS:diaglocal

@testset "hamiltonianmodels" for Ni = [1,2,3], Nj = [1,2,3]
    @test Ising(Ni,Nj) isa HamiltonianModel
    @test TFIsing(Ni,Nj,1.0) isa HamiltonianModel
    @test Heisenberg(Ni,Nj) isa HamiltonianModel
    @test diaglocal(Ni,Nj,[1.,-1]) isa HamiltonianModel
end