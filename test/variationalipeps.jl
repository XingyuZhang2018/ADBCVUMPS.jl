using Test
using ADBCVUMPS
using ADBCVUMPS: energy, num_grad, diaglocal
using OMEinsum, Zygote, Random
using Optim, LineSearches
using LinearAlgebra: svd, norm

@testset "non-interacting" for Ni = [2], Nj = [2]
    Random.seed!(100)
    rd = randn()
    rd2 = randn(2,2,2,2,2)
    model = diaglocal(Ni,Nj,[1,-1.0])
    h = hamiltonian(model)
    as = (reshape([rand(3,3,3,3,2) for i = 1:Ni*Nj],(Ni,Nj)) for _ in 1:10)
    @test all(a -> -1 < energy(h, model, SquareBCIPEPS(a); χ=5, tol=1e-10, maxiter=10)/2 < 1, as)

    a = reshape([zeros(2,2,2,2,2) .+ 1e-12 * rd2 for i = 1:Ni*Nj],(Ni,Nj))
    for j=1:Nj, i=1:Ni
        a[i,j][1,1,1,1,2] = rd
    end
    @test energy(h, model, SquareBCIPEPS(a); χ=4, tol=1e-10, maxiter=10)/2 ≈ -1

    a = reshape([zeros(2,2,2,2,2) .+ 1e-12 * rd2 for i = 1:Ni*Nj],(Ni,Nj))
    for j=1:Nj, i=1:Ni
        a[i,j][1,1,1,1,1] = rd
    end
    @test energy(h, model, SquareBCIPEPS(a); χ=4, tol=1e-10, maxiter=10)/2 ≈ 1

    a = reshape([zeros(2,2,2,2,2) .+ 1e-12 * rd2 for i = 1:Ni*Nj],(Ni,Nj))
    for j=1:Nj, i=1:Ni
        a[i,j][1,1,1,1,1] = a[i,j][1,1,1,1,2] = rd
    end
    @test abs(energy(h,model,SquareBCIPEPS(a); χ=4, tol=1e-10, maxiter=10)) < 1e-9

    grad = let energy = x -> real(energy(h, model, SquareBCIPEPS(reshape([x for i = 1:Ni*Nj],(Ni, Nj))); χ=4, tol=1e-10, maxiter=10))
        res = optimize(energy,
            Δ -> Zygote.gradient(energy,Δ)[1], a[1,1], LBFGS(m=20), inplace = false)
    end
    @test grad != Nothing

    hdiag = [0.3,-0.43]
    model = diaglocal(Ni,Nj,hdiag)
    bcipeps, key = init_ipeps(model; D=2, χ=4, tol=1e-10, maxiter=20)
    res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    e = minimum(res)/2
    @test isapprox(e, minimum(hdiag), atol=1e-3)
end

@testset "gradient" for Ni = [2], Nj = [2]
    Random.seed!(0)
    model = TFIsing(Ni,Nj,1.0)
    h = hamiltonian(model)
    bcipeps, key = init_ipeps(model; D=2, χ=4, tol=1e-10, maxiter=20)
    gradzygote = first(Zygote.gradient(bcipeps) do x
        energy(h,model,x; χ=4, tol=1e-10, maxiter=20)
    end).bulk
    gradnum = num_grad(bcipeps.bulk, δ=1e-3) do x
        energy(h,model,SquareBCIPEPS(x); χ=4, tol=1e-10, maxiter=20)
    end
    @test isapprox(gradzygote, gradnum, atol=1e-3)

    # Random.seed!(3)
    # model = Heisenberg(Ni,Nj)
    # h = hamiltonian(model)
    # bcipeps, key = init_ipeps(model; D=2, χ=4, tol=1e-10, maxiter=20)
    # gradzygote = first(Zygote.gradient(bcipeps) do x
    #     energy(h,model,x; χ=4, tol=1e-10, maxiter=20)
    # end).bulk
    # gradnum = num_grad(bcipeps.bulk, δ=1e-3) do x
    #     energy(h,model, SquareBCIPEPS(x); χ=4, tol=1e-10, maxiter=20)
    # end
    # @test isapprox(gradzygote , gradnum, atol=1e-3)
end

@testset "TFIsing" for Ni = [2], Nj = [2]
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(3)
    model = TFIsing(Ni,Nj,1.0)
    bcipeps, key = init_ipeps(model; D=2, χ=5, tol=1e-10, maxiter=20)
    res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    e = minimum(res)
    @test isapprox(e, -2.12566, atol = 1e-2)

    # Random.seed!(3)
    # model = TFIsing(Ni,Nj,0.5)
    # bcipeps, key = init_ipeps(model; D=2, χ=5, tol=1e-10, maxiter=20)
    # res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -2.0312, atol = 1e-2)

    # Random.seed!(3)
    # model = TFIsing(Ni,Nj,2.0)
    # bcipeps, key = init_ipeps(model; D=2, χ=5, tol=1e-10, maxiter=20)
    # res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -2.5113, atol = 1e-2)
end

@testset "heisenberg" for Ni = [2], Nj = [2]
    # comparison with results from https://github.com/wangleiphy/tensorgrad
    Random.seed!(100)
    model = Heisenberg(Ni,Nj,1.0,1.0,1.0)
    bcipeps, key = init_ipeps(model; D=3, χ=20, tol=1e-10, maxiter=20)
    res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    e = minimum(res)
    @test isapprox(e, -0.66023, atol = 1e-4)

    # Random.seed!(100)
    # model = Heisenberg(Ni,Nj,1.0,2.0,2.0)
    # bcipeps, key = init_ipeps(model; D=2, χ=5, tol=1e-10, maxiter=20)
    # res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -1.190, atol = 1e-3)

    # Random.seed!(100)
    # model = Heisenberg(Ni,Nj,2.0,0.5,0.5)
    # bcipeps, key = init_ipeps(model; D=2, χ=5, tol=1e-10, maxiter=20)
    # res = optimiseipeps(bcipeps, key; f_tol = 1e-6)
    # e = minimum(res)
    # @test isapprox(e, -1.0208, atol = 1e-3)
end