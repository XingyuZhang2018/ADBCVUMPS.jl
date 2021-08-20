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

energyplot = plot()
for D = 2:3
    χ = 20
    degree = 0.0:5.0:355.0
    yenergy = []
    folder = "./data/all/"
    for x in degree
        file = folder*"Kitaev_Heisenberg{Float64}($(x))_Array/Kitaev_Heisenberg{Float64}($(x))_Array_D$(D)_chi$(χ)_tol1.0e-10_maxiter10_miniter5.log"
        yenergy = [yenergy; read_low_energy(file)]
    end
    # degree = 0:pi/36:(2*pi-pi/36)
    # plot!(energyplot, degree, yenergy, xticks=(0.5 * pi .* (0:4), ["0", "π/2", "π", "3π/2", "2π"]), title = "energy-ϕ", label = "Stripy Ordered",legend = :bottomright, xlabel = "ϕ degree", ylabel = "E", lw = 3)
    plot!(energyplot, degree, yenergy, title = "energy-ϕ", label = "D = $(D)",legend = :bottomright, xlabel = "ϕ degree", ylabel = "E", lw = 2)
end
vline!(energyplot, [90, 180, 270])
savefig(energyplot,"./plot/Kitaev_Heisenberg-ϕ.svg")