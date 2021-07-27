using ADBCVUMPS
using ADBCVUMPS: buildbcipeps, energy, optcont
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
    real(energy(h, BCIPEPS{LT}(buildbcipeps(atype(x0),Ni,Nj)), oc, key; verbose=true))
end

model = Kitaev(-1.0,-1.0,-1.0)
bcipeps, key = init_ipeps(model; atype = CuArray, D=4, χ=20, tol=1e-10, maxiter=10)
x = []
yenergy = []
for χ in 55:5:80
    x = [x; χ]
    yenergy = [yenergy; energy_χ(bcipeps, key, χ)]
end
energyplot = plot()
yenergy = [-0.3933403183529031;
-0.3931582708269492;
-0.39301944939132616;
-0.392592776425696;
-0.39244779204734864;
-0.392474625166208;
-0.3924194840843681;
-0.39241896815255506;
-0.3924108901329133;
-0.3924003214385193;
-0.39239531646415066;
-0.39243941095717255;
-0.39239280014960354;
-0.3923914455125451;
-0.39238747530837353]
x = 10:5:80
plot!(energyplot, x, yenergy, title = "energy", label = "energy", lw = 3)