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
D = 4
    χ = 50
    ϕ = 0.01:0.01:0.08
    yenergy = []
    folder = "./../../../../data/xyzhang/ADBCVUMPS/K_Γ/"
    for x in ϕ
        file = folder*"K_Γ{Float64}($(x))_CuArray/D$(D)_chi$(χ)_tol1.0e-10_maxiter10_miniter2.log"
        yenergy = [yenergy; read_low_energy(file)]
    end
    # Γ = 0:pi/36:(2*pi-pi/36)
    # plot!(energyplot, Γ, yenergy, xticks=(0.5 * pi .* (0:4), ["0", "π/2", "π", "3π/2", "2π"]), title = "energy-ϕ", label = "Stripy Ordered",legend = :bottomright, xlabel = "ϕ Γ", ylabel = "E", lw = 3)
    plot!(energyplot, ϕ, yenergy, title = "energy-ϕ", label = "D = $(D)",legend = :bottomright, xlabel = "ϕ", ylabel = "E", lw = 2)

# vline!(energyplot, [270])
# savefig(energyplot,"./plot/K_Γ-ϕ.svg")