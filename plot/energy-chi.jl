using ADBCVUMPS
using ADBCVUMPS: buildbcipeps, energy, optcont
using CUDA
using Plots
CUDA.allowscalar(false)

# device!(0)
function new_energy(bulk, new_key)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = new_key
    hx, hy, hz = hamiltonian(model)
    h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    energy(h, buildbcipeps(atype(bulk),2,2), oc, new_key; verbose = true, savefile = true)
end

model = K_Γ(0.0)
folder = "./../../../../data1/xyzhang/ADBCVUMPS/K_Γ/"
field, atype, D, χ, tol, maxiter, miniter = 0.0, CuArray, 5, 50, 1e-10, 10, 2

bulk, _ = init_ipeps(model; folder = folder, atype = atype, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)

x = 50:10:80
yenergy = []
# ymag = []
for χ in x
    new_key = (folder, model, field, atype, D, χ, tol, maxiter, miniter)
    ener = new_energy(bulk, new_key)
    yenergy = [yenergy; ener]
    # ymag = [ymag; mag]
end
# yenergy /= 2
energyplot = plot()
# magplot = plot()
x
plot!(energyplot, x, real(yenergy), title = "energy", label = "energy",legend = :bottomright, xlabel = "χ", ylabel = "E", lw = 3)
# plot!(magplot, x, ymag, title = "magnetization", label = "magnetization",legend = :bottomright, xlabel = "χ", ylabel = "M", lw = 3)
# obs = plot(energyplot, magplot, layout = (2,1), xlabel="χ", size = [800, 1000])
# savefig(energyplot,"./plot/obs-$(model)_D$(D)_χ$(χ).svg")