using ADBCVUMPS
using ADBCVUMPS: buildbcipeps, energy, optcont
using CUDA
using Plots

function new_energy(bulk, new_key)
    folder, model, atype, D, χ, tol, maxiter, miniter = new_key
    hx, hy, hz = hamiltonian(model)
    h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    energy(h, buildbcipeps(atype(bulk),2,2), oc, new_key; verbose = true, savefile = true)
end

model = Kitaev_Heisenberg(270.0)
folder = "E:/1 - research/4.9 - AutoDiff/data/ADBCVUMPS.jl/Kitaev_Heisenberg/all/"
atype, D, χ, tol, maxiter, miniter = CuArray, 4, 20, 1e-10, 10, 2


bulk, _ = init_ipeps(model; folder = folder, atype = atype, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter)

x = 20:10:100
yenergy = []
# ymag = []
for χ in x
    new_key = (folder, model, atype, D, χ, tol, maxiter, miniter)
    ener = new_energy(bulk, new_key)
    yenergy = [yenergy; ener]
    # ymag = [ymag; mag]
end
yenergy
energyplot = plot()
# magplot = plot()
plot!(energyplot, x, yenergy, title = "energy", label = "energy",legend = :bottomright, xlabel = "χ", ylabel = "E", lw = 3)
# plot!(magplot, x, ymag, title = "magnetization", label = "magnetization",legend = :bottomright, xlabel = "χ", ylabel = "M", lw = 3)
# obs = plot(energyplot, magplot, layout = (2,1), xlabel="χ", size = [800, 1000])
# savefig(obs,"./plot/obs-χ_$(key).svg")