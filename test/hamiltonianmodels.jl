using Test
using BCVUMPS
using ADBCVUMPS
using ADBCVUMPS:HamiltonianModel

@testset "hamiltonianmodels" begin
    @test Ising() isa BCVUMPS.HamiltonianModel
    @test TFIsing(1.0) isa HamiltonianModel
    @test Heisenberg() isa HamiltonianModel
end