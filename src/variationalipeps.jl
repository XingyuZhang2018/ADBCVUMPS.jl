using BCVUMPS: bigleftenv, bigrightenv, ALCtoAC, obs2x2FL, obs2x2FR, leftenv, rightenv
using CUDA
using LinearAlgebra: I, norm
using LineSearches
using OMEinsum: get_size_dict, optimize_greedy,  MinSpaceDiff
using Optim
using TimerOutputs

function overlap(bulk, D, atype)
    ID = Matrix{Float64}(I, D, D)
    I2 = Matrix{Float64}(I, 2, 2)
    overlap11 = ein"ae, bf, cd -> abcdef"(ID, ID, I2)
    overlap11 = reshape(overlap11, D, D*2, D*2, D)
    overlap12 = permutedims(bulk, (5,1,2,3,4))
    overlap12 = reshape(overlap12, D*2, D, D, D)
    overlap21 = reshape(bulk, D, D, D, D*2)
    overlap22 = ein"ac, bd -> abcd"(ID, ID)
    reshape([atype(overlap11), atype(overlap21), atype(overlap12), atype(overlap22)], 2,2)
end

"""
    energy(h, bcipeps; χ, tol, maxiter)

return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, bulk, oc, key; verbose = false)
    bulk = symmetrize(bulk)
    model, atype, D, χ, tol, maxiter = key
    a = overlap(bulk, D, atype)
    Ni,Nj = size(bulk)
    bulk = atype(bulk)
    ap = ein"abcdx,ijkly -> aibjckdlxy"(bulk, conj(bulk))
    ap = reshape(ap, D^2, D^2, D^2, D^2, 2, 2)

    env = obs_bcenv_oneside(model, a; atype = atype, D = D, χ = χ, tol = tol, maxiter = maxiter, miniter = 10, verbose = verbose, savefile = true)
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
function expectationvalue(h, ap, env, oc)
    _, ALu, Cu, ARu, FL, FR = env
    χ, D, _ = size(ALu[1,1]) 
    oc1, oc2 = oc
    # a = reshape([CuArray(ein"ijklaa -> ijkl"(ap))],1,1)
    ap /= norm(ap)
    ap = CuArray(ap)

    BgALu = CuArray(reshape(ein"adb, bec -> adec"(ALu[1,1],ALu[1,2]), (χ, D^2, χ)))
    BgARu = CuArray(reshape(ein"adb, bec -> adec"(ARu[1,1],ARu[1,2]), (χ, D^2, χ)))
    
    BgFL = CuArray(reshape(ein"cde, abc -> abde"(FL[1,1],FL[2,1]), (χ, D^2, χ)))
    BgFR = CuArray(reshape(ein"abc, cde -> adbe"(FR[1,2],FR[2,2]), (χ, D^2, χ)))

    # BgALu = reshape([BgALu],1,1)
    # BgARu = reshape([BgARu],1,1)
    # BgFL = reshape([BgFL],1,1)
    # BgFR = reshape([BgFR],1,1)
    # _, BgFL = leftenv(BgALu, BgALu, a, BgFL)
    # _, BgFR = rightenv(BgARu, BgARu, a, BgFR)
    # BgFL = BgFL[1,1]
    # BgFR = BgFR[1,1]
    # BgALu = BgALu[1,1]
    # BgARu = BgARu[1,1]

    lr = oc1(BgFL,BgALu,ap,BgALu,CuArray(Cu[1,2]),CuArray(Cu[1,2]),BgFR,BgARu,ap,BgARu)
    e = ein"pqrs, pqrs -> "(lr,CuArray(h))
    n = ein"pprr -> "(lr)
    println("── = $(Array(e)[]/Array(n)[])") 
    etol = Array(e)[]/Array(n)[]

    # Zygote.@ignore begin
    #     _, BgFL1s, _= eigsolve(X->ein"(((dcba,def),ckge),bjhk),aji -> fghi"(X,BgALu,a,a,BgALd), rand(χ,D^2,D^2,χ), 1, :LM; ishermitian = false)
    #     BgFL = BgFL1s[1]
    #     _, BgFR1s, _= eigsolve(X->ein"(((fghi,def),ckge),bjhk),aji -> dcba"(X,BgARu,a,a,BgARd), rand(χ,D^2,D^2,χ), 1, :LM; ishermitian = false)
    #     BgFR = BgFR1s[1]
    #     lr2 = oc2(BgFL,BgALu,Cu[1,2],ap,ap,BgALd,Cd[1,2],BgFR)
    #     e2 = ein"pqrs, pqrs -> "(lr2,h)
    #     n2 = ein"pprr -> "(lr2)
    #     println("| = $(Array(e2)[]/Array(n2)[])") 
    # end

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
    f(x) = real(energy(h, x, oc, key; verbose=verbose))
    # ff(x) = real(energy(h, x, oc, key; verbose=verbose))
    g(x) = Zygote.gradient(f,atype(x))[1]
    res = optimize(f, g, 
        bulk, optimmethod, inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)), 
        )
    # println(to)
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