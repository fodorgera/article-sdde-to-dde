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
    # τt(t) = 1.5*π+(0.5*sin(t))
    τt(t) = 2*π
end;

# deterministic history φ(t) for t≤0
φ(t) = [0.0, 0.0];
T = 30;
sol1, L1 = solve_moments(A, B, c, α, β, γ; τ=τt, τmax=τ, T=T, φ=φ, depth=1, saveat=0:0.05:T);
sol2, L2 = solve_moments(A, B, c, α, β, γ; τ=τt, τmax=τ, T=T, φ=φ, depth=2, saveat=0:0.05:T);
# Use tighter tolerances and/or MethodOfSteps(Tsit5()) if avg+std still lags reference:
# using DelayDiffEq
sol10, L10 = solve_moments(A, B, c, α, β, γ; τ=τt, τmax=τ, T=T, φ=φ, depth=10, saveat=0:0.05:T,
    alg=MethodOfSteps(Tsit5()), reltol=1e-8, abstol=1e-10);
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
avgs = x_m1;
stds = sqrt.(x_m2 - avgs.^2);

plot(sol1.t, [avgs, avgs .+ stds], label=["avg" "avg + std"])

moments2 = [get_moments_at(sol2,L2,ti) for ti in sol2.t];
m12 = [m[1] for m in moments2];
m22 = [m[2] for m in moments2];
sk2 = [m[3] for m in moments2];
NT2 = length(sk2[1]);

x_m12 = [m12i[1] for m12i in m12];
x_m22 = [m22i[1,1] for m22i in m22];
x_sk2 = [[sk2i[1,1] for sk2i in [ski[i] for ski in sk2]] for i in 1:NT2];
avgs2 = x_m12;
stds2 = sqrt.(x_m22 - avgs2.^2);

plot!(sol2.t, [avgs2, avgs2 .+ stds2], label=["avg" "avg + std"])

moments3 = [get_moments_at(sol10,L10,ti) for ti in sol10.t];
m13 = [m[1] for m in moments3];
m23 = [m[2] for m in moments3];
sk3 = [m[3] for m in moments3];
NT3 = length(sk3[1]);

x_m13 = [m13i[1] for m13i in m13];
x_m23 = [m23i[1,1] for m23i in m23];
x_sk3 = [[sk3i[1,1] for sk3i in [ski[i] for ski in sk3]] for i in 1:NT3];
avgs3 = x_m13;
stds3 = sqrt.(x_m23 - avgs3.^2);

plot!(sol10.t, [avgs3, avgs3 .+ stds3], label=["avg" "avg + std"])

# verify with msdi
using MSDI

msdi_prob = MSDIProblem(A, B, c, α, β, γ, τ, τt);

u01 = [0.0, 0.0];
u02 = u01*u01';
msdi_sol = msdi_solve_opt(msdi_prob, undef; u01 = u01, u02 = u02, isTimeDependent = true, isTimeDependentDelay = true,method=:tr, tmax=T, k=2000);

msdi_vars = MSDI.getVar(msdi_sol[3]);
msdi_avgs = MSDI.getMean(msdi_sol[3]);

plot(msdi_avgs.ts, [msdi_avgs.values[1], msdi_avgs.values[1] .+ sqrt.(msdi_vars.values[1])], label=["avg" "avg + std"])

# verify with ensemble simulations
using DifferentialEquations
using StochasticDelayDiffEq
using DifferentialEquations.EnsembleAnalysis

function mathieu_model_f(du,u,h,p,t)
    uτ = h(p,t-τt(t))
    du .= A(t)*u .+ B(t)*uτ + c(t)
end
function mathieu_model_g(du,u,h,p,t)
    uτ = h(p,t-τt(t))
    du .= α(t)*u .+ β(t)*uτ + γ(t)
end

struct H_u0{T}
    u0::T
end
(h::H_u0)(p, t) = h.u0
ensemble_prob = EnsembleProblem(StochasticDelayDiffEq.SDDEProblem(mathieu_model_f, mathieu_model_g,u01,H_u0(u01),(0.0,T),[], constant_lags=[τ]));

ensemble_sol = solve(ensemble_prob, SRA3();dt=τ/500, trajectories=500);

# timeseries_point_meanvar returns (mean, variance); variance is E[x^2]-E[x]^2, so std = sqrt(variance)
sim_avgs, sim_vars = timeseries_point_meanvar(ensemble_sol, ensemble_sol[1].t);
sim_avgs_1 = [u[1] for u in sim_avgs.u];
sim_vars_1 = [u[1] for u in sim_vars.u];

sim_stds = sqrt.(sim_vars_1);

plot!(sim_avgs.t, [sim_avgs_1, sim_avgs_1 .+ sim_stds])