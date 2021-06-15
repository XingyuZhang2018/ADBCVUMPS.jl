using BCVUMPS: bigleftenv, bigrightenv, ALCtoAC, obs2x2FL, obs2x2FR
using LinearAlgebra: I, norm
using LineSearches
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Optim
using TimerOutputs

"""
    energy(h, bcipeps; χ, tol, maxiter)

return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, bcipeps::BCIPEPS, oc, key; verbose = false)
    model, atype, _, χ, tol, maxiter = key
    # bcipeps = indexperm_trisymmetrize(bcipeps)  # NOTE: this is not good
    D = getd(bcipeps)^2
    s = gets(bcipeps)
    bulk = bcipeps.bulk
    Ni,Nj = size(bulk)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D, D, D, D, s, s) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)

    rt = []
    folder = "./data/$(model)_$(atype)/"
    mkpath(folder)
    chkp_file = folder*"bcvumps_env_$(Ni)x$(Nj)_D$(D)_chi$(χ).jld2"
    if isfile(chkp_file)
        rt = SquareBCVUMPSRuntime(a, chkp_file, χ; verbose = verbose)
    else
        rt = SquareBCVUMPSRuntime(a, Val(:random), χ; verbose = verbose)
    end
    # @show typeof(rt)
    env = bcvumps(rt; tol=tol, maxiter=maxiter, verbose = verbose)
    Zygote.@ignore begin
        M, AL, C, AR, FL, FR = env.M, Array{Array{Float64,3},2}(env.AL), Array{Array{Float64,2},2}(env.C), Array{Array{Float64,3},2}(env.AR), Array{Array{Float64,3},2}(env.FL), Array{Array{Float64,3},2}(env.FR)
        envsave = SquareBCVUMPSRuntime(M, AL, C, AR, FL, FR)
        save(chkp_file, "env", envsave)
    end
    e = expectationvalue(h, ap, env, oc)
    return e
end

function optcont(D::Int, χ::Int)
    sd = Dict('n' => D^2, 'f' => χ, 'd' => D^2, 'e' => χ, 'o' => D^2, 'h' => χ, 'j' => χ, 'i' => D^2, 'k' => D^2, 'r' => 2, 's' => 2, 'q' => 2, 'a' => χ, 'c' => χ, 'p' => 2, 'm' => χ, 'g' => D^2, 'l' => χ, 'b' => D^2)
    oc1 = optimize_greedy(ein"abc,cde,bnodpq,anm,ef,ml,hij,fgh,okigrs,lkj -> pqrs", sd; method=MinSpaceDiff())
    sd = Dict('a' => χ, 'b' => D^2, 'c' => χ, 'd' => D^2, 'e' => D^2, 'f' => D^2, 'g' => D^2, 'h' => D^2, 'i' => χ, 'j' => D^2, 'k' => χ, 'r' => 2, 's' => 2, 'p' => 2, 'q' => 2, 'l' => χ, 'm' => χ)
    oc2 = optimize_greedy(ein"adgi,abl,lc,dfebpq,gjhfrs,ijm,mk,cehk -> pqrs", sd; method=MinSpaceDiff())
    oc1, oc2
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareBCVUMPSRuntime` `env`.
"""
function expectationvalue(h, ap, env::SquareBCVUMPSRuntime, oc)
    M,AL,C,AR,FL,FR = env.M,env.AL,env.C,env.AR,env.FL,env.FR
    oc1, oc2 = oc
    _, FL = obs2x2FL(AL, AR, M, FL)
    _, FR = obs2x2FR(AR, AL, M, FR)
    Ni,Nj = size(M)
    # Ni,Nj = 1,2
    ap /= norm(ap)
    etol = 0
    hx, hy, hz = h
    for j = 1:Nj, i = 1:Ni
        if (i,j) in [(1,1),(2,2)]
            hij = hy
        else
            hij = hx
        end
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc1(FL[i,j],AL[i,j],ap[i,j],permutedims(AR[ir,j],(3,2,1)),C[i,j],permutedims(C[ir,jr],(2,1)),FR[i,jr],AR[i,jr],ap[i,jr],permutedims(AL[ir,jr],(3,2,1)))
        e = ein"pqrs, pqrs -> "(lr,hij)
        n = ein"pprr -> "(lr)
        @show Array(e)[]/Array(n)[]
        etol += Array(e)[]/Array(n)[]
    end
    
    # AC = ALCtoAC(AL,C)
    _, BgFL = bigleftenv(AL, AR, M)
    _, BgFR = bigrightenv(AR, AL, M)
    for j = 1:Nj, i = 1:Ni
        if (i,j) in [(1,1),(2,2)]
            hij = hz
            ir = i + 1 - Ni * (i==Ni)
            jr = j - 1 + (j==1) * Nj
            # irr = i + 2 - Ni * (i + 2 > Ni)
            lr2 = oc2(BgFL[i,j],AL[i,j],C[i,j],ap[i,j],ap[ir,j],permutedims(AR[i,j],(3,2,1)),permutedims(C[i,jr],(2,1)),BgFR[i,j])
            e2 = ein"pqrs, pqrs -> "(lr2,hij)
            n2 = ein"pprr -> "(lr2)
            @show Array(e2)[]/Array(n2)[]
            etol += Array(e2)[]/Array(n2)[]
        end
    end

    return etol/Ni/Nj
end

function buildbcipeps(bulk)
    bulk = trisymmetrize(bulk)
    SquareBCIPEPS(reshape([bulk,permutedims(bulk, (3,4,1,2,5)),permutedims(bulk, (3,4,1,2,5)),bulk], (2, 2)))
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; atype = Array, D::Int, χ::Int, tol::Real, maxiter::Int, verbose = true)
    key = (model, atype, D, χ, tol, maxiter)
    folder = "./data/$(model)_$(atype)/"
    mkpath(folder)
    chkp_file = folder*"$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["bcipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        bulk = rand(D,D,D,D,2)
        verbose && println("random initial BCiPEPS $chkp_file")
    end
    bcipeps = buildbcipeps(bulk)
    # bcipeps = indexperm_trisymmetrize(bcipeps)
    return bcipeps, key
end

"""
    optimiseipeps(bcipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an bcipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(bcipeps::BCIPEPS{LT}, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20)) where LT
    model, atype, D, χ, _, _ = key
    hx, hy, hz = hamiltonian(model)
    h = (atype(hx),atype(hy),atype(hz))
    to = TimerOutput()
    oc = optcont(D, χ)
    f(x) = @timeit to "forward" real(energy(h, buildbcipeps(atype(x)), oc, key; verbose=verbose))
    ff(x) = real(energy(h, buildbcipeps(atype(x)), oc, key; verbose=verbose))
    g(x) = @timeit to "backward" Zygote.gradient(ff,atype(x))[1]
    x0 = bcipeps.bulk[1,1]
    res = optimize(f, g, 
        x0, optimmethod, inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
    println(to)
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `bcipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=2))   $(os.iteration)   $(os.value)   $(os.g_norm)\n"

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    model, atype, D, χ, tol, maxiter = key
    if !(key === nothing)
        logfile = open("./data/$(model)_$(atype)/$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save("./data/$(model)_$(atype)/$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "bcipeps", os.metadata["x"])
    end
    return false
end