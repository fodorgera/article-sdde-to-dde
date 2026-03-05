include("core_time_dep.jl")

begin
    # RA=0.25
    # RP=0.4
    ζ = 0.05
    H = 0.4
    σ_base = 0.1;
    σ = σ_base/H;
    Ω0 = 0.85
    # Ω0 = 0.87;
    RVA = 0.1
    RVF = 0.1
    Ω(t) = Ω0 * (1 + RVA * sin(RVF * t))
    # Ω(t) = Ω0
    ωm = π / 10
    # τ(t) = 2 * π / Ω(t)
    τ(t) = 2 * π / Ω0
    # τmax = 2 * π / (Ω0 * (1 - RVA))
    # τmax = τ(0.0);
    τmax = 2 * π / Ω0
end;

begin
    # turning example
    n = 2
    A(t) = [0 1.0; -(1 + H) -2ζ]
    B(t) = [0 0; H 0]
    c(t) = [0.0, 0.0]
    α(t) = [0 0.0; 0.0 0.0];
    β(t) = [0 0.0; 0.0 0.0];
    γ(t) = [0.0, σ*H]
end;

φ(t) = [0.01, 0.0];
T = 10*τmax;
sol1, L1 = solve_moments(A, B, c, α, β, γ;τmax=τmax, τ=τ, T=T, φ=φ, depth=1, saveat=0:0.05:T);
sol10, L10 = solve_moments(A, B, c, α, β, γ;τmax=τmax, τ=τ, T=T, φ=φ, depth=10, saveat=0:0.05:T);

using Plots

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