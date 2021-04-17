using ADBCVUMPS
using Test

@testset "ADBCVUMPS.jl" begin
    @testset "hamiltonianmodels" begin
        println("hamiltonianmodels tests running...")
        include("hamiltonianmodels.jl")
    end

    @testset "bcipeps" begin
        println("bcipeps tests running...")
        include("bcipeps.jl")
    end
    
end
