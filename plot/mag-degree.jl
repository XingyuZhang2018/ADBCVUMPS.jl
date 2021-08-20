using ADBCVUMPS
using ADBCVUMPS:σx,σy,σz,buildbcipeps
using BCVUMPS
using LinearAlgebra: norm
using OMEinsum
using Plots
using Random

function magnetisation(degree, folder, D, χ, tol, maxiter, miniter)
    bulk, key = init_ipeps(Kitaev_Heisenberg(degree); folder = folder, atype = Array, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
    folder, model, atype, D, χ, tol, maxiter, miniter = key
    Ni, Nj = 2, 2
    bulk = buildbcipeps(bulk,Ni,Nj)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 2, 2) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)

    env = obs_bcenv(model, a; atype = atype, D = D^2, χ = χ, tol = tol, maxiter = maxiter, miniter = miniter, verbose = false, savefile = false, folder = folder)

    M, ALu, Cu, _, ALd, Cd, _, FL, FR = env
    ap /= norm(ap)

    M = zeros(2,2)
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        lr3 = ein"(((((gea,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        Mx = ein"pq, pq -> "(Array(lr3),σx)
        My = ein"pq, pq -> "(Array(lr3),σy)
        Mz = ein"pq, pq -> "(Array(lr3),σz)
        n3 = ein"pp -> "(lr3)
        M[i,j] = ((Array(Mx)[]/Array(n3)[])^2+(Array(My)[]/Array(n3)[])^2+(Array(Mz)[]/Array(n3)[])^2)^0.5
    end
    mag = ((M[1,1]^2 + M[2,1]^2 + M[1,2]^2 + M[2,2]^2)/4)^0.5
    ferro = ((M[1,1] + M[2,1] + M[1,2] + M[2,2])^2/4)^0.5
    stripy = ((M[1,1] - M[2,1] - M[1,2] + M[2,2])^2/4)^0.5
    zigzag = ((M[1,1] + M[2,1] - M[1,2] - M[2,2])^2/4)^0.5
    Neel = ((M[1,1] - M[2,1] + M[1,2] - M[2,2])^2/4)^0.5
    return mag, ferro, stripy, zigzag, Neel
end

Random.seed!(100)
folder, D, χ, tol, maxiter, miniter = "/home/xyzhang/research/ADBCVUMPS.jl/data/all/", 3, 20, 1e-10, 10, 5

magplot = plot()
degree = 85.0:5.0:95.0
mag, ferro, stripy, zigzag, Neel = [], [], [], [], []
for x in degree
    y1, y2, y3, y4, y5 = magnetisation(x, folder, D, χ, tol, maxiter, miniter)
    mag = [mag; y1]
    ferro = [ferro; y2]
    stripy = [stripy; y3]
    zigzag = [zigzag; y4]
    Neel = [Neel; y5]
end
# plot!(magplot, degree, mag, title = "mag-ϕ", label = "mag D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "mag", lw = 2)
# plot!(magplot, degree, ferro, title = "mag-ϕ", label = "ferro D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "ferro", lw = 2)
plot!(magplot, degree, stripy, title = "mag-ϕ", label = "stripy D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "stripy", lw = 2)
plot!(magplot, degree, zigzag, title = "mag-ϕ", label = "zigzag D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "zigzag", lw = 2)
plot!(magplot, degree, Neel, title = "mag-ϕ", label = "Neel D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "Neel", lw = 2)
