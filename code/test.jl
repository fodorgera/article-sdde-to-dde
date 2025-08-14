include("./core.jl");

# Problem size
n = 2
A = [-0.8 1.0;
    -1.0 -0.4]
B = [0.1 0.0;
    0.0 0.2]
c = [0.0, 0.0]

# multiplicative noise (scalar Wiener)
α = 0.3I(n)
β = 0.1I(n)
γ = [0.2, 0.0]  # affine diffusion offset

τ = 0.7
T = 5.0

# deterministic history φ(t) for t≤0
φ(t) = [sin(0.5 * t) + 1.0,
    cos(0.3 * t)]

sol, L = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T)

# extract at a time
μt, Mt, St = get_moments_at(sol, L, 5.0)
println("μ(5.0) = ", μt)
println("M(5.0) = \n", Mt)
println("Number of cross moments S_k tracked: ", length(St))
println("S₁(5.0) = \n", St[1])
