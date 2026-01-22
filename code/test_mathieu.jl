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
end;

# deterministic history φ(t) for t≤0
φ(t) = [0.0, 0.0];
T = 10*τ;
sol1, L1 = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, depth=1, saveat=0:0.05:T);
sol2, L2 = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, depth=2, saveat=0:0.05:T);
sol10, L10 = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, depth=10, saveat=0:0.05:T);
# extract at a time
# plot at tspan
using Plots

# moments
moments1 = [get_moments_at(sol1,L1,ti) for ti in sol1.t];
m1 = [m[1] for m in moments1];
m2 = [m[2] for m in moments1];
sk = [m[3] for m in moments1];
NT = length(sk[1]);

x_m1 = [m1i[1] for m1i in m1];
x_m2 = [m2i[1,1] for m2i in m2];
x_sk = [[sk1i[1,1] for sk1i in [ski[i] for ski in sk]] for i in 1:NT];

plot(sol1.t, x_m1, label="m1")
plot(sol1.t, x_m2, label="m2")
plot(sol1.t, x_sk[1], label="sk1")

moments2 = [get_moments_at(sol2,L2,ti) for ti in sol2.t];
m12 = [m[1] for m in moments2];
m22 = [m[2] for m in moments2];
sk2 = [m[3] for m in moments2];
NT2 = length(sk2[1]);

x_m12 = [m12i[1] for m12i in m12];
x_m22 = [m22i[1,1] for m22i in m22];
x_sk2 = [[sk2i[1,1] for sk2i in [ski[i] for ski in sk2]] for i in 1:NT2];

plot!(sol2.t, x_m12, label="m1")
plot!(sol2.t, x_m22, label="m2")
plot!(sol2.t, x_sk2[1], label="sk1")
plot(sol2.t, x_sk2[2], label="sk2")

moments3 = [get_moments_at(sol10,L10,ti) for ti in sol10.t];
m13 = [m[1] for m in moments3];
m23 = [m[2] for m in moments3];
sk3 = [m[3] for m in moments3];
NT3 = length(sk3[1]);

x_m13 = [m13i[1] for m13i in m13];
x_m23 = [m23i[1,1] for m23i in m23];
x_sk3 = [[sk3i[1,1] for sk3i in [ski[i] for ski in sk3]] for i in 1:NT3];

plot!(sol10.t, x_m13, label="m1")
plot!(sol10.t, x_m23, label="m2")
plot!(sol10.t, x_sk3[1], label="sk1")
plot(sol10.t, x_sk3[2], label="sk2")