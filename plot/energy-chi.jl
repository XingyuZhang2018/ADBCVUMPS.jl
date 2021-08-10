using ADBCVUMPS
using ADBCVUMPS: buildbcipeps, energy, optcont
using CUDA
using Plots

function energy_χ(bcipeps::BCIPEPS{LT}, key, χ) where LT
    model, atype, D, _, tol, maxiter = key
    key = (model, atype, D, χ, tol, maxiter)
    # h = atype(hamiltonian(model))
    hx, hy, hz = hamiltonian(model)
    h = (atype(hx),atype(hy),atype(hz))
    Ni, Nj = 2, 2
    oc = optcont(D, χ)
    x0 = zeros(D,D,D,D,2,2)
    x0[:,:,:,:,:,1] = bcipeps.bulk[1,1]
    x0[:,:,:,:,:,2] = bcipeps.bulk[2,1]
    energy(h, BCIPEPS{LT}(buildbcipeps(atype(x0),Ni,Nj)), oc, key; verbose=true)
end

model = Kitaev(-1.0,-1.0,-1.0)
bcipeps, key = init_ipeps(model; atype = CuArray, D=4, χ=20, tol=1e-10, maxiter=10)
x = 10:5:80
yenergy = []
ymag = []
for χ in x
    ener, mag = energy_χ(bcipeps, key, χ)
    yenergy = [yenergy; ener]
    ymag = [ymag; mag]
end
energyplot = plot()
magplot = plot()
plot!(energyplot, x, yenergy, title = "energy", label = "energy",legend = :bottomright, xlabel = "χ", ylabel = "E", lw = 3)
plot!(magplot, x, ymag, title = "magnetization", label = "magnetization",legend = :bottomright, xlabel = "χ", ylabel = "M", lw = 3)
obs = plot(energyplot, magplot, layout = (2,1), xlabel="χ", size = [800, 1000])
savefig(obs,"./plot/obs-χ_$(key).svg")