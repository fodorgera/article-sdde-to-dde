using Revise
using SddeToOde
using LinearAlgebra
using DifferentialEquations
using Plots
# Problem size
n = 1;
A = -6.0I(n);
B = 0.0*I(n);
c = [0.0];

# multiplicative noise (scalar Wiener)
α = 0.0I(n);
β = 2.0I(n);
γ = [1.0];  # affine diffusion offset

τ = 1.0;
T = 2.0;

# deterministic history φ(t) for t≤0
φ(t) = [3.0];

hayes_ode, hayes_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ);

hayes_sol = solve(hayes_ode, Tsit5(), dt=τ/1000, adaptive=false);

hayes_res = [SddeToOde.get_x_moments(hayes_sol, hayes_meta, t) for t in hayes_sol.t];

avgs = [r[1][1] for r in hayes_res];
vars = [r[2][1,1] for r in hayes_res];

plot(hayes_sol.t, [avgs, avgs .+ sqrt.(vars)], label=["avg" "avg + std"])

# validation with MSDI
using MSDI
msdi_prob = MSDIProblem(t->A, t->B, t->c, t->α, t->β, t->γ, τ,t->τ);

u01 = [3.0];
u02 = u01*u01';
msdi_sol = msdi_solve_opt(msdi_prob, undef; u01 = u01, u02 = u02, isTimeDependent = false, isTimeDependentDelay = false ,method=:ssm, tmax=T, k=1000);

msdi_vars = MSDI.getVar(msdi_sol[3]);
msdi_avgs = MSDI.getMean(msdi_sol[3]);
msdi_stds = sqrt.(msdi_vars.values[1]);

plot!(msdi_avgs.ts, [msdi_avgs.values[1], msdi_avgs.values[1] .+ msdi_stds], label=["avg" "avg + std"])