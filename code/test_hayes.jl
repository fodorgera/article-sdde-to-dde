include("core_time_dep.jl")

using LinearAlgebra

# Problem size
n = 1;
A = -6.0I(n);
B = 0.0I(n);
c = [0.0];

# multiplicative noise (scalar Wiener)
α = 0.0I(n);
β = 2I(n);
γ = [1.0];  # affine diffusion offset

τ = 1.0;
T = 5.0;

# deterministic history φ(t) for t≤0
φ(t) = [3.0];
sol, L = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T);
# extract at a time
μt, Mt, St = get_moments_at(sol, L, 0.0);

# plot at tspan
using Plots
plot(sol, idxs=[2], xlabel="t", ylabel="μ(t)", title="Mean μ(t) over time")

vars = [sqrt(s[2] - s[1]^2) for s in sol.u];
avgs = [s[1] for s in sol.u];
ts = sol.t;

plot(ts, [avgs, avgs .+ vars])