using ADBCVUMPS
using ADBCVUMPS:σx,σy,σz,buildbcipeps
using BCVUMPS
using CUDA
using FileIO
using LinearAlgebra: norm
using OMEinsum
using Plots
using Random

function magnetisation(model, folder, D, χ, tol, maxiter, miniter)
    bulk, key = init_ipeps(model; folder = folder, atype = CuArray, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = key
    Ni, Nj = 2, 2
    bulk = buildbcipeps(bulk,Ni,Nj)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 2, 2) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)
    
    chkp_file_obs = folder*"obs_D$(D^2)_chi$(χ).jld2"
    FL, FR = load(chkp_file_obs)["env"]
    chkp_file_up = folder*"up_D$(D^2)_χ$(χ).jld2"                     
    rtup = SquareBCVUMPSRuntime(a, chkp_file_up, χ; verbose = false)   
    ALu,ARu,Cu = rtup.AL, rtup.AR, rtup.C
    chkp_file_down = folder*"down_D$(D^2)_χ$(χ).jld2"                              
    rtdown = SquareBCVUMPSRuntime(a, chkp_file_down, χ; verbose = false)   
    ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

    M = Array{Array{ComplexF64,1},2}(undef, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        lr3 = ein"(((((aeg,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        Mx = ein"pq, pq -> "(Array(lr3),σx/2)
        My = ein"pq, pq -> "(Array(lr3),σy/2)
        Mz = ein"pq, pq -> "(Array(lr3),σz/2)
        n3 = ein"pp -> "(lr3)
        M[i,j] = [Array(Mx)[]/Array(n3)[], Array(My)[]/Array(n3)[], Array(Mz)[]/Array(n3)[]]
        print("M[[$(i),$(j)]] = {")
        for k = 1:3 
            print(real(M[i,j][k])) 
            k == 3 ? println("};") : print(",")
        end
        # println("M = $(sqrt(real(M[i,j]' * M[i,j])))")
    end
    mag = (norm(M[1,1]) + norm(M[2,1]) + norm(M[2,2]) + norm(M[1,2]))/4
    ferro = norm((M[1,1] + M[1,2] + M[2,2] + M[2,1])/4)
    stripy = norm((M[1,1] - M[1,2] - M[2,2] + M[2,1])/4)
    zigzag = norm((M[1,1] + M[1,2] - M[2,2] - M[2,1])/4)
    Neel = norm((M[1,1] - M[1,2] + M[2,2] - M[2,1])/4)
    return mag, ferro, stripy, zigzag, Neel
end

Random.seed!(100)
folder, D, χ, tol, maxiter, miniter = "./../../../../data1/xyzhang/ADBCVUMPS/K_Γ/", 4, 30, 1e-10, 10, 2
ϕ = 0.0:0.005:0.035
magplot = plot()
mag, ferro, stripy, zigzag, Neel = [], [], [], [], []
for x in ϕ
    model = K_Γ(x)
    y1, y2, y3, y4, y5 = magnetisation(model, folder, D, χ, tol, maxiter, miniter)
    mag = [mag; y1]
    ferro = [ferro; y2]
    stripy = [stripy; y3]
    zigzag = [zigzag; y4]
    Neel = [Neel; y5]
end
plot!(magplot, ϕ, mag, shape = :circle, title = "mag-ϕ", label = "mag D = $(D)",legend = :inside, xlabel = "ϕ degree", lw = 2)
plot!(magplot, ϕ, ferro, title = "mag-ϕ", label = "ferro D = $(D)",legend = :inside, xlabel = "ϕ degree", ylabel = "ferro", lw = 2)
plot!(magplot, ϕ, stripy, title = "mag-ϕ", label = "stripy D = $(D)",legend = :inside, xlabel = "ϕ degree", ylabel = "stripy", lw = 2)
plot!(magplot, ϕ, zigzag, shape = :cross, title = "mag-ϕ", label = "zigzag D = $(D)",legend = :bottomright, xlabel = "ϕ degree", lw = 2)
plot!(magplot, ϕ, Neel, shape = :diamond, title = "mag-ϕ", label = "Neel D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "Order Parameters", lw = 2)
# savefig(magplot,"./plot/K_Γ_E-ϕ-D$(D)_χ$(χ).svg")
# @show zigzag, stripy, ferro, mag, Neel