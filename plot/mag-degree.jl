using ADBCVUMPS
using ADBCVUMPS:σx,σy,σz,buildbcipeps
using BCVUMPS
using CUDA
using FileIO
using LinearAlgebra: norm
using OMEinsum
using Plots
using Random

function magnetisation(degree, folder, D, χ, tol, maxiter, miniter)
    bulk, key = init_ipeps(Kitaev_Heisenberg(degree); folder = folder, atype = CuArray, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
    folder, model, atype, D, χ, tol, maxiter, miniter = key
    Ni, Nj = 2, 2
    bulk = buildbcipeps(bulk,Ni,Nj)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 2, 2) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)
    
    chkp_file_obs = folder*"$(model)_$(atype)/obs_D$(D^2)_chi$(χ).jld2"
    FL, FR = load(chkp_file_obs)["env"]
    chkp_file_up = folder*"$(model)_$(atype)/up_D$(D^2)_chi$(χ).jld2"                     
    rtup = SquareBCVUMPSRuntime(a, chkp_file_up, χ; verbose = false)   
    ALu,ARu,Cu = rtup.AL, rtup.AR, rtup.C
    chkp_file_down = folder*"$(model)_$(atype)/down_D$(D^2)_chi$(χ).jld2"                              
    rtdown = SquareBCVUMPSRuntime(a, chkp_file_down, χ; verbose = false)   
    ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

    M = zeros(2,2)
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        lr3 = ein"(((((gea,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        Mx = ein"pq, pq -> "(Array(lr3),σx/2)
        My = ein"pq, pq -> "(Array(lr3),σy/2)
        Mz = ein"pq, pq -> "(Array(lr3),σz/2)
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
folder, D, χ, tol, maxiter, miniter = "/home/xyzhang/research/ADBCVUMPS.jl/data/all/", 4, 20, 1e-10, 10, 2

magplot = plot()
degree = [266.0:2.0:268.0; 272.0:2.0:274.0]
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
