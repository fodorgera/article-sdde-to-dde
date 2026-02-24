# Quick check: layout and vec/unvec round-trip
using LinearAlgebra
n, K = 2, 2
i = 1
idx_μ = i:(i+n-1); i += n
idx_M = i:(i+n*n-1); i += n*n
idx_S = [i:(i+n*n-1) for _ in 1:K]
total = length(idx_μ) + length(idx_M) + K*n*n
@assert total == n + n*n + K*n*n
println("Layout: n=$n, K=$K -> state length $total")

# vec/reshape round-trip (column-major)
M = [1.0 2; 3 4]
v = vec(M)
M2 = reshape(v, n, n)
@assert M == M2
println("vec(M) then reshape(n,n) recovers M: OK")

# Simulate unpack_state then pack
y = zeros(total)
y[idx_μ] .= [0.1, 0.2]
y[idx_M] .= vec(M)
for k in 1:K
    y[idx_S[k]] .= vec(M .+ k)  # dummy S[k]
end
μ = @view y[idx_μ]
M_read = reshape(@view(y[idx_M]), n, n)
S_read = [reshape(@view(y[idx_S[k]]), n, n) for k in 1:K]
@assert μ == [0.1, 0.2]
@assert M_read == M
println("Pack then unpack recovers: OK")
