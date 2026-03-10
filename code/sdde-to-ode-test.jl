using Revise
using SddeToOde
using LinearAlgebra
using DifferentialEquations
using Plots
using Profile
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

Ae, ce, h = SddeToOde.build_extended_drift(A, B, c, τ; m=100);

hayes_ode, hayes_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=50);
hayes_dde, hayes_dde_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=100, dde=true);

GC.gc();
Profile.clear();
@time hayes_sol = solve(hayes_ode, Tsit5(), dt=τ/1000, adaptive=false, save_everystep=false, save_start=true, save_end=true);
Profile.print();
@time hayes_dde_sol = solve(hayes_dde, MethodOfSteps(Tsit5()), dt=τ/1000, adaptive=false);

hayes_res = [SddeToOde.get_x_moments(hayes_sol, hayes_meta, t) for t in hayes_sol.t];
hayes_dde_res = [SddeToOde.get_x_moments(hayes_dde_sol, hayes_dde_meta, t) for t in hayes_dde_sol.t];

avgs = [r[1][1] for r in hayes_res];
vars = [r[2][1,1] for r in hayes_res];

hayes_dde_avgs = [r[1][1] for r in hayes_dde_res];
hayes_dde_vars = [r[2][1,1] for r in hayes_dde_res];

plot(hayes_sol.t, avgs, label="moment ode")

plot(hayes_sol.t, [avgs, avgs .+ sqrt.(vars)], label=["avg" "avg + std"])
plot(hayes_dde_sol.t, [hayes_dde_avgs, hayes_dde_avgs .+ sqrt.(hayes_dde_vars)], label=["avg" "avg + std"])

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

plot(msdi_vars.ts, msdi_vars.values[1], label="msdi var")
plot!(hayes_sol.t, vars, label="moment ode")

# OSCILLATOR TEST

# problem size
n = 2;
p = 7; d = 3.5;
σ0 = 0.1;
A = [0.0 1.0; 5.0 0.0];
# A = [0.0 1.0; -5.0 0.0];
B = [0.0 0.0; -p -d];
# B = [0.0 0.0; 0.0 0.0];
β = σ0*B;
γ = [0.0, σ0];
τ = 0.3;
c = [0.0, 0.0];
α = [0.0 0.0; 0.0 0.0];

φ(t) = [0.1, 0.0];

T = 3.0;

osc_ode, osc_meta = SddeToOde.get_ode_from_sdde(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, m=20);

osc_sol = solve(osc_ode, Tsit5(), dt=τ/1000, adaptive=false);

osc_res = [SddeToOde.get_x_moments(osc_sol, osc_meta, t) for t in osc_sol.t];

osc_avgs = [r[1][1] for r in osc_res];
osc_vars = [r[2][1,1] for r in osc_res];

plot(osc_sol.t, [osc_avgs, osc_avgs .+ sqrt.(osc_vars)], label=["avg" "avg + std"])

# msdi validation
osc_msdi_prob = MSDIProblem(t->A, t->B, t->c, t->α, t->β, t->γ, τ, t->τ);
osc_msdi_sol = msdi_solve_opt(osc_msdi_prob, undef; u01 = [0.1, 0.0], u02 = [0.1, 0.0]*[0.1, 0.0]', isTimeDependent = false, isTimeDependentDelay = false, method=:ssm, tmax=T, k=1000);

osc_msdi_avgs = MSDI.getMean(osc_msdi_sol[3]);
osc_msdi_vars = MSDI.getVar(osc_msdi_sol[3]);
osc_msdi_stds = sqrt.(osc_msdi_vars.values[1]);

plot!(osc_msdi_avgs.ts, [osc_msdi_avgs.values[1], osc_msdi_avgs.values[1] .+ osc_msdi_stds], label=["avg" "avg + std"])

# MATHIEU TEST
begin
    a1 = 0.2
    δ = 2.0
    b0 = -0.2
    ε = 1.0
    c0 = 0.15
    σ0 = 0.2
    ω = 1.0
end
# Problem size
begin
    n = 2
    AM(t) = [0 1.0; -(δ + ε * cos(ω * t)) -a1]
    BM(t) = [0 0; b0 0]
    cM(t) = [0.0, c0]

    αM(t) = [0 0.0; -σ0*(δ+ε*cos(ω * t)) -σ0*a1]
    βM(t) = σ0 * BM(t)
    γM(t) = [0.0, σ0]

    τ = 2 * π
end;

# deterministic history φ(t) for t≤0
φ(t) = [0.0, 0.0];
T = 10;

mathieu_ode, mathieu_meta = SddeToOde.get_ode_from_sdde(AM, BM, cM, αM, βM, γM; τ=τ, T=T, φ=φ, m=10);

mathieu_sol = solve(mathieu_ode, Tsit5(), dt=τ/1000, adaptive=false);

mathieu_res = [SddeToOde.get_x_moments(mathieu_sol, mathieu_meta, t) for t in mathieu_sol.t];

mathieu_avgs = [r[1][1] for r in mathieu_res];
mathieu_vars = [r[2][1,1] for r in mathieu_res];

plot(mathieu_sol.t, [mathieu_avgs, mathieu_avgs .+ sqrt.(mathieu_vars)], label=["avg" "avg + std"])

# MSDI validation
mathieu_msdi_prob = MSDIProblem(AM, BM, cM, αM, βM, γM, τ, τt);

mathieu_msdi_sol = msdi_solve_opt(mathieu_msdi_prob, undef; u01 = [0.0, 0.0], u02 = [0.0, 0.0]*[0.0, 0.0]', isTimeDependent = true, isTimeDependentDelay = false, method=:ssm, tmax=T, k=1000);

mathieu_msdi_avgs = MSDI.getMean(mathieu_msdi_sol[3]);
mathieu_msdi_vars = MSDI.getVar(mathieu_msdi_sol[3]);
mathieu_msdi_stds = sqrt.(mathieu_msdi_vars.values[1]);

plot!(mathieu_msdi_avgs.ts, [mathieu_msdi_avgs.values[1], mathieu_msdi_avgs.values[1] .+ mathieu_msdi_stds], label=["avg" "avg + std"])