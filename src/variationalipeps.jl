using Optim, LineSearches
using LinearAlgebra: I, norm
using TimerOutputs

"""
    energy(h, bcipeps; χ, tol, maxiter)

return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, model::HamiltonianModel, bcipeps::BCIPEPS; χ::Int, tol::Real, maxiter::Int, verbose = false)
    bcipeps = indexperm_symmetrize(bcipeps)  # NOTE: this is not good
    D = getd(bcipeps)^2
    s = gets(bcipeps)
    Ni,Nj = size(bcipeps.bulk)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bcipeps.bulk[i], conj(bcipeps.bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D, D, D, D, s, s) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)

    folder = "./data/$(model)/"
    mkpath(folder)
    chkp_file = folder*"bcvumps_env_$(Ni)x$(Nj)_D$(D)_chi$(χ).jld2"
    if isfile(chkp_file)
        rt = SquareBCVUMPSRuntime(a, chkp_file, χ; verbose = verbose)
    else
        rt = SquareBCVUMPSRuntime(a, Val(:random), χ; verbose = verbose)
    end

    env = bcvumps(rt; tol=tol, maxiter=maxiter, verbose = verbose)
    save(chkp_file, "env", env)
    e = expectationvalue(h, ap, env)
    return e
end

"""
    expectationvalue(h, ap, env)

return the expectationvalue of a two-site operator `h` with the sites
described by rank-6 tensor `ap` each and an environment described by
a `SquareBCVUMPSRuntime` `env`.
"""
function expectationvalue(h, ap, env::SquareBCVUMPSRuntime)
    M,AL,C,AR,FL,FR = env.M,env.AL,env.C,env.AR,env.FL,env.FR
    Ni,Nj = size(M)
    # Ni,Nj = 1,1
    ap /= norm(ap)
    etol = 0
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i == Ni)
        jr = j + 1 - Nj * (j == Nj)
        e = ein"abc,cde,anm,ef,ml,fgh,lkj,hij,bnodpq,okigrs,pqrs -> "(FL[i,j],AL[i,j],conj(AL[ir,j]),C[i,j],conj(C[ir,j]),AR[i,jr],conj(AR[ir,jr]),FR[i,jr],ap[i,j],ap[i,jr],h)[]
        n = ein"abc,cde,anm,ef,ml,fgh,lkj,hij,bnodpq,okigrs -> pqrs"(FL[i,j],AL[i,j],conj(AL[ir,j]),C[i,j],conj(C[ir,j]),AR[i,jr],conj(AR[ir,jr]),FR[i,jr],ap[i,j],ap[i,jr])
        n = ein"pprr -> "(n)[]
        etol += e/n
    end
    
    # AC = ein"asc,cb -> asb"(AL,C)
    # _, FL4 = bigleftenv(AL, M)
    # _, FR4 = bigrightenv(AL, M)
    # e2 = ein"dcba,def,aji,fghi,ckgepq,bjhkrs,pqrs -> "(FL4,AC,conj(AC),FR4,ap,ap,h)[]
    # n2 = ein"dcba,def,aji,fghi,ckgepq,bjhkrs -> pqrs"(FL4,AC,conj(AC),FR4,ap,ap)
    # n2 = ein"pprr -> "(n2)[]
    # @show e/n e2/n2 (e/n+e2/n2)/2

    return etol/Ni/Nj
end

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)

Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int, verbose = true)
    folder = "./data/$(model)/"
    mkpath(folder)
    key = (model, D, χ, tol, maxiter)
    chkp_file = folder*"$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["bcipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        bulk = rand(D,D,D,D,2,2)
        verbose && println("random initial iPEPS")
    end
    Ni, Nj = model.Ni, model.Nj
    bcipeps = SquareBCIPEPS(reshape([bulk for i = 1:Ni*Nj],(Ni, Nj)))
    bcipeps = indexperm_symmetrize(bcipeps)
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
function optimiseipeps(bcipeps::BCIPEPS{LT}, key; f_tol = 1e-6, verbose= false, optimmethod = LBFGS(m = 20)) where LT
    model, D, χ, tol, maxiter = key
    h = hamiltonian(model)
    Ni, Nj = model.Ni, model.Nj
    to = TimerOutput()
    f(x) = @timeit to "forward" real(energy(h, model, BCIPEPS{LT}(reshape([x for i = 1:Ni*Nj],(Ni, Nj))); χ=χ, tol=tol, maxiter=maxiter, verbose=verbose))
    ff(x) = real(energy(h, model, BCIPEPS{LT}(reshape([x for i = 1:Ni*Nj],(Ni, Nj))); χ=χ, tol=tol, maxiter=maxiter, verbose=verbose))
    g(x) = @timeit to "backward" Zygote.gradient(ff,x)[1]
    res = optimize(f, g, 
        bcipeps.bulk[1,1], optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol,
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

    model, D, χ, tol, maxiter = key
    if !(key === nothing)
        logfile = open("./data/$(model)/$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).log", "a")
        write(logfile, message)
        close(logfile)
        save("./data/$(model)/$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2", "bcipeps", os.metadata["x"])
    end
    return false
end