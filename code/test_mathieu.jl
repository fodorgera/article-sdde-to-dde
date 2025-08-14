include("core_time_dep.jl")

using LinearAlgebra

begin
    a1 = 0.2
    δ = 2.0
    b0 = -0.2
    ε = 1.0
    c0 = 0.15
    σ0 = 0.3
    ω = 1.0
end
# Problem size
begin
    n = 2
    A(t) = [0 1.0; -(δ + ε * cos(ω * t)) -a1]
    B(t) = [0 0; b0 0]
    c(t) = [0.0, c0]

    # multiplicative noise (scalar Wiener)
    α(t) = [0 0.0; -σ0*(δ+ε*cos(ω * t)) -σ0*a1]
    β(t) = σ0 * B(t)
    γ(t) = [0.0, σ0]  # affine diffusion offset

    τ = 2 * π
    T = 2 * τ
end

# deterministic history φ(t) for t≤0
φ(t) = [0.0, 0.0];
sol, L = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T);
# extract at a time
μt, Mt, St = get_moments_at(sol, L, 0.0);

# plot at tspan
using Plots
plot(sol, idxs=[1], xlabel="t", ylabel="μ(t)", title="Mean μ(t) over time")

vars = [sqrt(abs(s[3] - s[1] * s[1])) for s in sol.u];
avgs = [s[1] for s in sol.u];
ts = sol.t;

plot(ts, [avgs, avgs .+ vars])