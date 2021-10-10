using ADBCVUMPS
using ADBCVUMPS:σx,σy,σz,buildbcipeps, optcont
using BCVUMPS
using CUDA
using FileIO
using LinearAlgebra: norm, I, cross
using OMEinsum
using Plots
using Random
using Statistics: std

function read_xy(file)
    f = open(file, "r" )
    n = countlines(f)
    seekstart(f)
    X = zeros(n)
    Y = zeros(n)
    for i = 1:n
        x,y = split(readline(f), ",")
        X[i],Y[i] = parse(Float64,x),parse(Float64,y)
    end
    X,Y
end

function observable(model, f, folder, D, χ, tol, maxiter, miniter)
    bulk, key = init_ipeps(model, [1.0,1.0,1.0], f; folder = folder, atype = CuArray, D=D, χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
    folder, model, field, atype, D, χ, tol, maxiter, miniter = key
    h = hamiltonian(model)
    oc = optcont(D, χ)
    Ni = 1
    Nj = Int(size(bulk,6) / Ni)
    bulk = buildbcipeps(bulk,Ni,Nj)
    ap = [ein"abcdx,ijkly -> aibjckdlxy"(bulk[i], conj(bulk[i])) for i = 1:Ni*Nj]
    ap = [reshape(ap[i], D^2, D^2, D^2, D^2, 4, 4) for i = 1:Ni*Nj]
    ap = reshape(ap, Ni, Nj)
    a = [ein"ijklaa -> ijkl"(ap[i]) for i = 1:Ni*Nj]
    a = reshape(a, Ni, Nj)
    
    chkp_file_obs = folder*"obs_D$(D^2)_chi$(χ).jld2"
    FL, FR = load(chkp_file_obs)["env"]
    chkp_file_up = folder*"up_D$(D^2)_χ$(χ).jld2"                     
    rtup = SquareBCVUMPSRuntime(a, chkp_file_up, χ; verbose = false)   
    FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
    chkp_file_down = folder*"down_D$(D^2)_χ$(χ).jld2"                              
    rtdown = SquareBCVUMPSRuntime(a, chkp_file_down, χ; verbose = false)   
    ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C

    M = Array{Array{ComplexF64,1},2}(undef, Nj, 2)
    Sx1 = reshape(ein"ab,cd -> acbd"(σx/2, I(2)), (4,4))
    Sx2 = reshape(ein"ab,cd -> acbd"(I(2), σx/2), (4,4))
    Sy1 = reshape(ein"ab,cd -> acbd"(σy/2, I(2)), (4,4))
    Sy2 = reshape(ein"ab,cd -> acbd"(I(2), σy/2), (4,4))
    Sz1 = reshape(ein"ab,cd -> acbd"(σz/2, I(2)), (4,4))
    Sz2 = reshape(ein"ab,cd -> acbd"(I(2), σz/2), (4,4))
    etol = 0.0
    for j = 1:Nj, i = 1:Ni
        jr = j + 1 - (j==Nj)*Nj
        ir = Ni + 1 - i
        lr3 = ein"(((((aeg,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        Mx1 = ein"pq, pq -> "(Array(lr3),Sx1)
        Mx2 = ein"pq, pq -> "(Array(lr3),Sx2)
        My1 = ein"pq, pq -> "(Array(lr3),Sy1)
        My2 = ein"pq, pq -> "(Array(lr3),Sy2)
        Mz1 = ein"pq, pq -> "(Array(lr3),Sz1)
        Mz2 = ein"pq, pq -> "(Array(lr3),Sz2)
        n3 = ein"pp -> "(lr3)
        M[j,1] = [Array(Mx1)[]/Array(n3)[], Array(My1)[]/Array(n3)[], Array(Mz1)[]/Array(n3)[]]
        M[j,2] = [Array(Mx2)[]/Array(n3)[], Array(My2)[]/Array(n3)[], Array(Mz2)[]/Array(n3)[]]
        print("M[[$(j),$(1)]] = {")
        for k = 1:3 
            print(real(M[j,1][k])) 
            k == 3 ? println("};") : print(",")
        end
        print("M[[$(j),$(2)]] = {")
        for k = 1:3 
            print(real(M[j,2][k])) 
            k == 3 ? println("};") : print(",")
        end
        if field != 0.0
            etol -= (real(M[j,1] + M[j,2]))' * field / 2
        end
        # println("M = $(sqrt(real(M[i,j]' * M[i,j])))")
    end
    Cross = norm(cross(M[1,1],M[1,2]))
    @show Cross
    mag = (norm(M[1,1]) + norm(M[2,1]) + norm(M[2,2]) + norm(M[1,2]))/4
    ferro = norm((M[1,1] + M[1,2] + M[2,2] + M[2,1])/4)
    zigzag = norm((M[1,1] + M[1,2] - M[2,2] - M[2,1])/4)
    stripy = norm((M[1,1] - M[1,2] + M[2,2] - M[2,1])/4)
    Neel = norm((M[1,1] - M[1,2] - M[2,2] + M[2,1])/4)

    oc1, oc2 = oc
    hx, hy, hz = h
    ap /= norm(ap)
    hx = reshape(permutedims(hx, (1,3,2,4)), (4,4))
    hy = reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hy, I(2)), (4,4,4,4))
    hz = reshape(ein"ae,bfcg,dh -> abefcdgh"(I(2), hz, I(2)), (4,4,4,4))

    Ex, Ey, Ez = 0, 0, 0
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc1(FL[i,j],ALu[i,j],ap[i,j],ALd[ir,j],Cu[i,j],Cd[ir,j],FR[i,jr],ARu[i,jr],ap[i,jr],ARd[ir,jr])
        ey = ein"pqrs, pqrs -> "(lr,hy)
        n = ein"pprr -> "(lr)
        Ey += Array(ey)[]/Array(n)[]
        println("hy = $(Array(ey)[]/Array(n)[])")
        etol += Array(ey)[]/Array(n)[]

        lr2 = ein"(((((aeg,abc),cd),ehfbpq),ghi),ij),dfj -> pq"(FL[i,j],ALu[i,j],Cu[i,j],ap[i,j],ALd[ir,j],Cd[ir,j],FR[i,j])
        ex = ein"pq, pq -> "(lr2,hx)
        n = Array(ein"pp -> "(lr2))[]
        Ex += Array(ex)[]/n
        println("hx = $(Array(ex)[]/n)")
        etol += Array(ex)[]/n
    end
    
    for j = 1:Nj, i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        lr3 = oc2(ALu[i,j],Cu[i,j],FLu[i,j],ap[i,j],FRu[i,j],FL[ir,j],ap[ir,j],FR[ir,j],ALd[i,j],Cd[i,j])
        ez = ein"pqrs, pqrs -> "(lr3,hz)
        n = ein"pprr -> "(lr3)
        Ez += Array(ez)[]/Array(n)[]
        println("hz = $(Array(ez)[]/Array(n)[])") 
        etol += Array(ez)[]/Array(n)[]
    end
    # ΔE = real(Ex - (Ey + Ez)/2)
    ΔE = std(real.([Ex, Ey, Ez]))
    @show ΔE

    println("e = $(etol/Ni/Nj)")
    return mag, ferro, stripy, zigzag, Neel, real(etol/Ni/Nj), ΔE, Cross
end

function deriv_y(x,y)
    n = length(x)
    dy = zeros(n)
    for i = 1:n
        if i == 1
            dy[i] = (y[i+1] - y[i])/(x[i+1] - x[i])
        elseif i == n
            dy[i] = (y[i] - y[i-1])/(x[i] - x[i-1])
        else
            dy[i] = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])
        end
    end
    return dy
end

Random.seed!(100)
folder, D, χ, tol, maxiter, miniter = "./../../../../data/xyzhang/ADBCVUMPS/K_J_Γ_Γ′_1x2/", 4, 80, 1e-10, 10, 2
f = [0.0;0.1;0.15:0.01:0.22;0.3:0.1:0.8]
# Γ = 0.3
mag, ferro, stripy, zigzag, Neel, E, ΔE, Cross = [], [], [], [], [], [], [], []
for x in f
    model = K_J_Γ_Γ′(-1.0, -0.1, 0.3, -0.02)
    y1, y2, y3, y4, y5, y6, y7, y8 = observable(model, x, folder, D, χ, tol, maxiter, miniter)
    mag = [mag; y1]
    ferro = [ferro; y2]
    stripy = [stripy; y3]
    zigzag = [zigzag; y4]
    Neel = [Neel; y5]
    E = [E; y6]
    ΔE = [ΔE; y7]
    Cross = [Cross; y8]
end

magplot = plot()
plot!(magplot, f, mag, shape = :auto, title = "mag-h", label = "mag D = $(D)", lw = 2)
plot!(magplot, f, ferro, shape = :auto, label = "ferro D = $(D)", lw = 2)
plot!(magplot, f, stripy, shape = :auto, label = "stripy D = $(D)", lw = 2)
plot!(magplot, f, zigzag, shape = :auto, label = "zigzag D = $(D)", lw = 2)
plot!(magplot, f, Neel, shape = :auto, label = "Neel D = $(D)",legend = :topright, xlabel = "h", ylabel = "Order Parameters", lw = 2)
dferro = deriv_y(f, ferro)
plot!(magplot, f, dferro, shape = :auto, label = "dferro D = $(D)", lw = 2)
# X,Y = read_xy(folder*"2019Gordon-mag.log")
# plot!(magplot, X, Y, shape = :auto, label = "2019Gordon 5° mag", lw = 2)
# X,Y = read_xy(folder*"2019Gordon-dmag.log")
# plot!(magplot, X, Y, shape = :auto, label = "2019Gordon 5° dmag",legend = :topright, xlabel = "h", ylabel = "Order Parameters",rightmargin = 1.5Plots.cm, lw = 2)


# Eplot = plot()
# plot!(Eplot, f, E, shape = :auto, label = "E 1x2 cell D = $(D)",legend = :topright, xlabel = "f", ylabel = "E", lw = 2)
# dEplot = twinx()
# dE = deriv_y(f, E)
# plot!(dEplot, f, dE, shape = :auto, label = "dE 1x2 cell D = $(D)",legend = :topright, xlabel = "f", ylabel = "dE", lw = 2)


# ΔEplot = plot()
# ΔEplot = twinx()
# plot!(ΔEplot, f, abs.(ΔE), shape = :x, label = "ΔE D = $(D)",legend = :bottomright, xlabel = "h", ylabel = "ΔE", lw = 2)

# Crossplot = plot()
# plot!(Crossplot, Γ, Cross, shape = :auto, label = "Cross D = $(D)",legend = :bottomright, xlabel = "Γ/|K|", ylabel = "Cross norm", lw = 2)
# savefig(Eplot,"./plot/K_Γ_1x2&1x3_E-Γ-D$(D)_χ$(χ).svg")
# @show zigzag stripy ferro mag Neel ΔE Cross