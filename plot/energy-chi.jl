using ADBCVUMPS
using ADBCVUMPS: energy, optcont, symmetrize, expectationvalue
using CUDA
using FileIO
using OMEinsum
using Plots
CUDA.allowscalar(false)

function energy_χ(bulk, key, χ)
    folder, model, atype, D, _, tol, maxiter, miniter = key
    key = (folder, model, atype, D, χ, tol, maxiter, miniter)
    h = atype(hamiltonian(model))
    # hx, hy, hz = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    chkp_file_up = folder*"$(model)_$(atype)/up_D$(D)_chi$(χ).jld2"
    # if isfile(chkp_file_up)
    #     env = load(chkp_file_up)["env"]
    #     env = env.M, env.AL, env.C, env.AR, env.FL, env.FR
    #     bulk = symmetrize(bulk)
    #     ap = ein"abcdx,ijkly -> aibjckdlxy"(bulk, conj(bulk))
    #     ap = reshape(ap, D^2, D^2, D^2, D^2, 2, 2)
    #     expectationvalue(h, ap, env, oc)
    # else
        real(energy(h, bulk, oc, key; verbose=true, savefile = true))
    # end
end

function energy_iter(bulk, key, iter)
    folder, model, atype, D, χ, tol, maxiter, miniter = key
    key = (folder, model, atype, D, χ, tol, iter, iter)
    h = atype(hamiltonian(model))
    # hx, hy, hz = hamiltonian(model)
    # h = (atype(hx),atype(hy),atype(hz))
    oc = optcont(D, χ)
    chkp_file_up = folder*"$(model)_$(atype)/up_D$(D)_chi$(χ).jld2"
    if isfile(chkp_file_up)
        env = load(chkp_file_up)["env"]
        bulk = symmetrize(bulk)
        ap = ein"abcdx,ijkly -> aibjckdlxy"(bulk, conj(bulk))
        ap = reshape(ap, D^2, D^2, D^2, D^2, 2, 2)
        expectationvalue(h, ap, env, oc)
    else
        real(energy(h, bulk, oc, key; verbose=true, savefile = false))
    end
end

model = Heisenberg(1, 1, 1.0,1.0,1.0)
D, χ = 2, 20
folder = "E:/1 - research/4.9 - AutoDiff/data/ADBCVUMPS.jl/Heisenberg-fold/"
bulk, key = init_ipeps(model; atype = Array, folder = folder, D=D, χ=χ, tol=1e-10, maxiter=10, miniter=2)
x = 20:10:50
yenergy = []
for iter in x
    global yenergy = [yenergy; energy_χ(bulk, key, iter)]
end
energyplot = plot()
plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)
# folder = "/home/xyzhang/research/ADBCVUMPS.jl/"
# savefig(energyplot,folder*"plot/obs-iter_D$(D)_χ$(χ).svg")
# yenergy
# bulk, key = init_ipeps(model; atype = CuArray, D=D, χ=χ, tol=1e-10, maxiter=50, miniter=50)
# x = 50:10:80
# yenergy = []
# for iter in x
#     global yenergy = [yenergy; energy_iter(bulk, key, iter)]
# end
# energyplot = plot()
# plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)