using ADBCVUMPS
using ADBCVUMPS: energy, optcont
using CUDA
using Plots

function energy_χ(bulk, key, χ)
    folder, model, atype, D, _, tol, maxiter, miniter = key
    key = (folder, model, atype, D, χ, tol, maxiter, miniter)
    h = atype(hamiltonian(model))
    # hx, hy, hz = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    @time real(energy(h, bulk, oc, key; verbose=true))
end

function energy_iter(bulk, key, iter)
    folder, model, atype, D, χ, tol, maxiter, miniter = key
    key = (folder, model, atype, D, χ, tol, iter, iter)
    h = atype(hamiltonian(model))
    # hx, hy, hz = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    @time real(energy(h, bulk, oc, key; verbose=true, savefile = false))
end

model = Heisenberg(1, 1, 1.0,1.0,1.0)
D, χ = 8, 160
folder = "/home/xyzhang/research/ADBCVUMPS.jl/data/"
bulk, key = init_ipeps(model; atype = CuArray, folder = folder, D=D, χ=χ, tol=1e-10, maxiter=50, miniter=50)
x = 120:10:180
yenergy = []
for iter in x
    global yenergy = [yenergy; energy_χ(bulk, key, iter)]
end
energyplot = plot()
plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)
folder = "/home/xyzhang/research/ADBCVUMPS.jl/"
savefig(energyplot,folder*"plot/obs-iter_D$(D)_χ$(χ).svg")

# bulk, key = init_ipeps(model; atype = CuArray, D=D, χ=χ, tol=1e-10, maxiter=50, miniter=50)
# x = 50:10:80
# yenergy = []
# for iter in x
#     global yenergy = [yenergy; energy_iter(bulk, key, iter)]
# end
# energyplot = plot()
# plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)