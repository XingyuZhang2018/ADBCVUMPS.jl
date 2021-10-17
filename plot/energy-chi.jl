using ADBCVUMPS
using ADBCVUMPS: buildbcipeps, energy, optcont
using CUDA
using Plots
CUDA.allowscalar(false)

device!(7)
function new_energy(bulk, new_key)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = new_key
    h = hamiltonian(model)
    oc = optcont(D, χ)
    Ni,Nj = 1,2
    energy(h, buildbcipeps(atype(bulk),Ni,Nj), oc, new_key; verbose = true, savefile = true)
end

function init_ipeps_type(model::HamiltonianModel, fdirection::Vector{Float64} = [0.0,0.0,0.0], field::Float64 = 0.0, type::String = "ferro"; folder::String="./data/", atype = Array, D::Int, χ::Int, tol::Real, maxiter::Int, miniter::Int, verbose = true)
    if field == 0.0
        folder *= "$(model)/"
    else
        folder *= "$(model)_field$(fdirection)_$(field)$(type)/"
        field = field * fdirection / norm(fdirection)
    end
    mkpath(folder)
    chkp_file = folder*"D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter)_miniter$(miniter).jld2"
    if isfile(chkp_file)
        bulk = load(chkp_file)["bcipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        bulk = rand(ComplexF64,D,D,D,D,4,2)
        verbose && println("random initial BCiPEPS $chkp_file")
    end
    bulk /= norm(bulk)
    key = (folder, model, field, atype, D, χ, tol, maxiter, miniter)
    return bulk, key
end

model = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
for field in 0.79
    folder = "./../../../../data/xyzhang/ADBCVUMPS/K_J_Γ_Γ′_1x2/"
    fdirection, atype, D, χ, tol, maxiter, miniter = [1.0,1.0,0.825221], CuArray, 4, 80, 1e-10, 10, 2

    bulk, key = init_ipeps_type(model, fdirection, field, ""; folder = folder, atype = atype, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = key

    x = 80

    yenergy = []
    # ymag = []
    for χ in x
        new_key = (folder, model, field, atype, D, χ, tol, 10, 2)
        ener = new_energy(bulk, new_key)
        yenergy = [yenergy; ener]
        # ymag = [ymag; mag]
    end
end
# yenergy /= 2
# energyplot = plot()
# magplot = plot()
# plot!(energyplot, x, real(yenergy), title = "energy", label = "energy",legend = :bottomright, xlabel = "χ", ylabel = "E", lw = 3)
# plot!(magplot, x, ymag, title = "magnetization", label = "magnetization",legend = :bottomright, xlabel = "χ", ylabel = "M", lw = 3)
# obs = plot(energyplot, magplot, layout = (2,1), xlabel="χ", size = [800, 1000])
# savefig(energyplot,"./plot/obs-$(model)_D$(D)_χ$(χ).svg")