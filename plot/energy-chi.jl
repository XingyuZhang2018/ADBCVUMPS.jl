using ADBCVUMPS
using ADBCVUMPS: buildbcipeps, energy, optcont
using CUDA
using Plots
CUDA.allowscalar(false)

device!(0)
function new_energy(bulk, new_key)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = new_key
    h = hamiltonian(model)
    oc = optcont(D, χ)
    Ni,Nj = 1,3
    energy(h, buildbcipeps(atype(bulk),Ni,Nj), oc, new_key; verbose = true, savefile = true)
end

for Γ in 0.2
    model = K_J_Γ_Γ′(-1.0, 0.0, Γ, 0.0)
    folder = "./../../../../data/xyzhang/ADBCVUMPS/K_J_Γ_Γ′_1x3/"
    field, atype, D, χ, tol, maxiter, miniter = 0.0, CuArray, 5, 100, 1e-10, 10, 2

    bulk, key = init_ipeps(model; folder = folder, atype = atype, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = key

    yenergy = []
    # ymag = []
    for χ in 100
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