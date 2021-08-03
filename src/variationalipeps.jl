using BCVUMPS: bigleftenv, bigrightenv, ALCtoAC, obs2x2FL, obs2x2FR
using LinearAlgebra: I, norm
using LineSearches
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Optim
using TimerOutputs

function overlap(bulk, D)
    ID = Matrix{Float64}(I, D, D)
    I2 = Matrix{Float64}(I, 2, 2)
    overlap11 = ein"ae, bf, cd -> abcdef"(ID, ID, I2)
    overlap11 = reshape(overlap11, D, D*2, D^2, D)
    overlap12 = permutedims(bulk, (5,1,2,3,4))
    overlap12 = reshape(overlap12, D*2, D, D, D)
    overlap21 = reshape(bulk, D, D, D, D*2)
    overlap22 = ein"ac, bd -> abcd"(ID, ID)
    reshape([overlap11, overlap21, overlap12, overlap22], 2,2)
end

"""
    energy(h, bcipeps; χ, tol, maxiter)

return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, bulk, oc, key; verbose = false)
    bulk = symmetrize(bulk)
    model, atype, D, χ, tol, maxiter = key
    a = overlap(bulk, D)
    Ni,Nj = size(bulk)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(bulk, conj(bulk))
    ap = reshape(ap, D^2, D^2, D^2, D^2, 2, 2)

    env = obs_bcenv(model, a; atype = atype, D = D^2, χ = χ, tol = tol, maxiter = maxiter, miniter = 5, verbose = verbose, savefile = true)
    e = expectationvalue(h, ap, env, oc)
    return e
end

function optcont(D::Int, χ::Int)
    sd = Dict('n' => D^4, 'f' => χ, 'd' => D^4, 'e' => χ, 'o' => D^4, 'h' => χ, 'j' => χ, 'i' => D^4, 'k' => D^4, 'r' => 2, 's' => 2, 'q' => 2, 'a' => χ, 'c' => χ, 'p' => 2, 'm' => χ, 'g' => D^4, 'l' => χ, 'b' => D^4)
    oc1 = optimize_greedy(ein"abc,cde,bnodpq,anm,ef,ml,hij,fgh,okigrs,lkj -> pqrs", sd; method=MinSpaceDiff())
    sd = Dict('a' => χ, 'b' => D^4, 'c' => χ, 'd' => D^4, 'e' => D^4, 'f' => D^4, 'g' => D^4, 'h' => D^4, 'i' => χ, 'j' => D^4, 'k' => χ, 'r' => 2, 's' => 2, 'p' => 2, 'q' => 2, 'l' => χ, 'm' => χ)
    oc2 = optimize_greedy(ein"adgi,abl,lc,dfebpq,gjhfrs,ijm,mk,cehk -> pqrs", sd; method=MinSpaceDiff())
    oc1, oc2
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareBCVUMPSRuntime` `env`.
"""
function expectationvalue(h, ap, env, oc)
    M, ALu, Cu, ARu, ALd, Cd, ARd, _, _ = env
    χ, D, _ = size(ALu[1,1]) 
    oc1, oc2 = oc
    Ni,Nj = size(M)
    ap /= norm(ap)

    _, BgFL = bigleftenv(ALu, ALd, M)
    _, BgFR = bigrightenv(ARu, ARd, M)
    FL = reshape(BgFL[1,1], χ, D^2, χ)
    FR = reshape(BgFR[1,2], χ, D^2, χ)

    BgALu = reshape(ein"adb, bec -> adec"(ALu[1,1],ALu[1,2]), (χ, D^2, χ))
    BgARu = reshape(ein"adb, bec -> adec"(ARu[1,1],ARu[1,2]), (χ, D^2, χ))
    BgALd = reshape(ein"adb, bec -> adec"(ALd[1,1],ALd[1,2]), (χ, D^2, χ))
    BgARd = reshape(ein"adb, bec -> adec"(ARd[1,1],ARd[1,2]), (χ, D^2, χ))
    
    lr = oc1(FL,BgALu,ap,BgALd,Cu[1,2],Cd[1,2],FR,BgARu,ap,BgARd)
    e = ein"pqrs, pqrs -> "(lr,h)
    n = ein"pprr -> "(lr)
    println("── = $(Array(e)[]/Array(n)[])") 
    etol = Array(e)[]/Array(n)[]
    
    return etol
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
        bulk = load(chkp_file)["ipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        bulk = rand(D,D,D,D,2)
        verbose && println("random initial BCiPEPS $chkp_file")
    end
    bulk = symmetrize(bulk)
    return bulk, key
end

"""
    optimiseipeps(bcipeps, h; χ, tol, maxiter, optimargs = (), optimmethod = LBFGS(m = 20))

return the tensor `bulk'` that describes an bcipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimiseipeps(bulk, key; f_tol = 1e-6, opiter = 100, verbose= false, optimmethod = LBFGS(m = 20)) where LT
    model, atype, D, χ, _, _ = key
    h = atype(hamiltonian(model))
    Ni, Nj = 2, 2
    to = TimerOutput()
    oc = optcont(D, χ)
    f(x) = @timeit to "forward" real(energy(h, x, oc, key; verbose=verbose))
    ff(x) = real(energy(h, x, oc, key; verbose=verbose))
    g(x) = @timeit to "backward" Zygote.gradient(ff,atype(x))[1]
    res = optimize(f, g, 
        bulk, optimmethod, inplace = false,
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
        save("./data/$(model)_$(atype)/$(model)_$(atype)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "ipeps", os.metadata["x"])
    end
    return false
end