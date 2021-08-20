using ADBCVUMPS
using ADBCVUMPS: energy, optcont
using CUDA
using Plots

function energy_χ(bulk, key, χ)
    model, atype, D, _, tol, maxiter = key
    key = (model, atype, D, χ, tol, maxiter)
    h = atype(hamiltonian(model))
    # hx, hy, hz = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    @time real(energy(h, bulk, oc, key; verbose=true))
end

model = Heisenberg(1, 1, 1.0,1.0,1.0)
bulk, key = init_ipeps(model; atype = CuArray, D=6, χ=80, tol=1e-10, maxiter=10)
x = 80
yenergy = []
for χ in x
    yenergy = [yenergy; energy_χ(bulk, key, χ)]
end
# energyplot = plot()
# plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)