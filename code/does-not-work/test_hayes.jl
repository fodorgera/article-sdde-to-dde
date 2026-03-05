include("core_time_dep.jl")

include("core_manual.jl")
using LinearAlgebra
using Plots

# Problem size
n = 1;
# A = -6.0I(n);
A = 0.0*I(n)
# B = 3.0I(n);
B = 1.0*I(n)
c = [0.0];

# multiplicative noise (scalar Wiener)
α = 0.0I(n);
β = 0.0I(n);
γ = [1.0];  # affine diffusion offset

τ = 1.0;
T = 2.0;

# deterministic history φ(t) for t≤0
φ(t) = [3.0];

sol, L = solve_moments_manual(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T);

sol_c, L_c = solve_centered_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T, depth=5);

sol_dl, L_dl = solve_centered_moments_delayline(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T, depth=3, m=5);

avgs_dl = [s.x[1][1] for s in sol_dl.u];
vars_dl = [s.x[2][1,1] for s in sol_dl.u];
stds_dl = sqrt.(abs.(vars_dl));
plot(sol_dl.t, vars_dl, label="moment dde delayline")

avgs_c = [s.x[1][1] for s in sol_c.u];
vars_c = [s.x[2][1,1] for s in sol_c.u];
stds_c = sqrt.(abs.(vars_c));
plot(sol_c.t, [avgs_c, avgs_c .+ stds_c], label=["avg" "avg + std"])
plot!(sol_c.t, vars_c, label="moment dde")

sol, L = solve_extended_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T);

res = [get_x_moments(sol, L, t) for t in sol.t];

avgs = [r[1][1] for r in res];
vars = [r[2][1,1] for r in res];

plot(sol.t, avgs, label="avg")
plot(sol.t, vars, label="moment dde")

plot(sol.t,[sol.u[i][2] for i in 1:length(sol.u)])
plot!(sol.t,[sol.u[i][1] for i in 1:length(sol.u)])

sol, L = solve_moments(A, B, c, α, β, γ; τ=τ, T=T, φ=φ, saveat=0:0.05:T, depth = 1);
# extract at a time
μt, Mt, St = get_moments_at(sol, L, 0.0);

# plot at tspan
using Plots
plot(sol, idxs=[2], xlabel="t", ylabel="μ(t)", title="Mean μ(t) over time")

vars = [sqrt(s[2] - s[1]^2) for s in sol.u];
avgs = [s[1] for s in sol.u];
ts = sol.t;
m2 = [s[2] for s in sol.u];

test_avgs = sqrt.(abs.(m2));

plot(ts, [avgs, test_avgs], label=["avg" "avg + std"])

# validate with msdi
using MSDI
msdi_prob = MSDIProblem(t->A, t->B, t->c, t->α, t->β, t->γ, τ,t->τ);

u01 = [3.0];
u02 = u01*u01';
msdi_sol = msdi_solve_opt(msdi_prob, undef; u01 = u01, u02 = u02, isTimeDependent = false, isTimeDependentDelay = false ,method=:ssm, tmax=T, k=1000);

msdi_vars = MSDI.getVar(msdi_sol[3]);
msdi_avgs = MSDI.getMean(msdi_sol[3]);
msdi_stds = sqrt.(msdi_vars.values[1]);

plot!(msdi_avgs.ts, [msdi_avgs.values[1], msdi_avgs.values[1] .+ msdi_stds], label=["avg" "avg + std"])
plot!(msdi_vars.ts, msdi_vars.values[1], label="MSDI var")