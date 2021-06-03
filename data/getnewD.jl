using Base: CodegenParams
using ADBCVUMPS
using JLD2
using FileIO

function get_new_D(folder, model, params, nparams)
     D,  χ,  tol,  maxiter = params
    nD, nχ, ntol, nmaxiter = nparams
    if nD>D
        d = 2
        old_chkp_file = folder*"$(model)_D$(D)_chi$(χ)_tol$(tol)_maxiter$(maxiter).jld2"
        bulk = load(old_chkp_file)["bcipeps"]
        nbulk = zeros(nD, nD, nD, nD, d, 2)
        s = 1:D
        nbulk[s,s,s,s,1:d,1:2] = bulk
        s = D+1:nD
        for i=s,j=s,k=s,l=s,m=1:d,n=1:2
            nbulk[i,j,k,l,m,n] = 0.0
        end
        new_chkp_file = folder*"$(model)_D$(nD)_chi$(nχ)_tol$(ntol)_maxiter$(nmaxiter).jld2"

        save(new_chkp_file, "bcipeps", nbulk)
        printstyled("created new file: "*new_chkp_file*"\n"; bold=true, color=:red)
    end
end

Ni,Nj = 2,2
model = Heisenberg(Ni,Nj,1.0,1.0,1.0)
folder = "./data/$(model)/"

params = 3, 20, 1e-10, 20
nparams = 4, 30, 1e-10, 20
get_new_D(folder, model, params, nparams)