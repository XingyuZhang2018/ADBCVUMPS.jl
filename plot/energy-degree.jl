using Plots

function read_low_energy(file)
    f = open(file, "r" )
    n = countlines(f)
    seekstart(f)
    energy = zeros(n)
    for i = 1:n
        _, _, e, _ = split(readline(f), "   ")
        energy[i] = parse(Float64,e)
    end
    minimum(energy)
end

degree = -55.0:-5.0:-90.0
yenergy = []
for x in degree
    file = "./data/Kitaev_Heisenberg{Float64}($(x))_Array/Kitaev_Heisenberg{Float64}($(x))_Array_D2_chi20_tol1.0e-10_maxiter10_miniter5.log"
    global yenergy = [yenergy; read_low_energy(file)]
end
energyplot = plot()
plot!(energyplot, degree, yenergy, title = "energy", label = "energy",legend = :bottomright, xlabel = "χ", ylabel = "E", lw = 3)