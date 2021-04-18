using BenchmarkTools

function foo1()
    s = 0 
    for i in 1:1e5
        s += i
    end 
end

@benchmark foo1()