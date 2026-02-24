# include("core_time_dep.jl")
include("core_manual.jl")

using LinearAlgebra

# problem size
n = 2;
p = 7; d = 3.5;
σ0 = 0.0;
A = [0.0 1.0; 5.0 0.0];
# A = [0.0 1.0; -5.0 0.0];
B = [0.0 0.0; -p -d];
# B = [0.0 0.0; 0.0 0.0];
β = σ0*B;
γ = [0.0, σ0];
τ = 0.3;
c = [0.0, 0.0];
α = [0.0 0.0; 0.0 0.0];

T = 3.0;

φ(t) = [0.1, 0.0];
sol, L = solve_moments_manual(A, B, c, α, β, γ; τmax=τ, τ=τ, T=T, φ=φ, saveat=0:0.05:T);

get_moments_at(sol, L, T)
sol.u[1]
using Plots
plot([s[1] for s in sol.u], xlabel="t", ylabel="μ(t)", title="Mean μ(t) over time")

sol.u[61].x[2]

function getVars(sol, L)
    vars = [];
    for s in sol.u
        μt = s.x[1]
        # Mt = s.x[2]
        S1 = s.x[3];
        # push!(vars, (Mt - μt*μt')[1,1]);
        push!(vars, S1[1,1]);
    end
    return vars
end

vars = getVars(sol, L);

vars

plot(sol.t, vars, xlabel="t", ylabel="var(t)", title="Variance var(t) over time")

u01 = φ(0.0);
u02 = u01*u01';

# msdi
T = 3.;
using MSDI
msdi_prob = MSDIProblem(t->A, t->B, t->c, t->α, t->β, t->γ, τ, t->τ);
msdi_sol = msdi_solve_opt(msdi_prob, undef; u01 = u01, u02 = u02, isTimeDependent = false, isTimeDependentDelay = false, method=:ssm, tmax=T, k=1000);
msdi_vars = MSDI.getVar(msdi_sol[3]);
msdi_avgs = MSDI.getMean(msdi_sol[3]);
plot!(msdi_vars.ts, msdi_vars.values[1], xlabel="t", ylabel="var(t)", title="Variance var(t) over time")
plot!(msdi_avgs.ts, [msdi_avgs.values[1], msdi_avgs.values[1] .+ sqrt.(msdi_vars.values[1])], label=["avg" "avg + std"])