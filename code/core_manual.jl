using DifferentialEquations
using RecursiveArrayTools

"""
Manual moment system with explicit state components (no layout packing).
State is `u = (μ::Vector, M::Matrix, S₁::Matrix, S₂::Matrix)` in an `ArrayPartition`.
S₁ = E[x(t)x(t-τ)'], S₂ = E[x(t)x(t-2τ)']. S₃ is closed as μ(t)μ(t-3τ)'.
Returns just `sol`; use `get_moments_manual(sol, t)` to read out (μ, M, S₁, S₂).
"""
function solve_moments_manual(A, B, c, α, β, γ; τ, T, φ,
                              τmax=τ, tspan=(0.0, T), saveat=nothing, kwargs...)
    n = size(A, 1)
    τ0 = τmax

    # Collect parameters we may want to tweak/extend later
    p = (A=A, B=B, c=c, α=α, β=β, γ=γ, τ=τ0, φ=φ)

    # History for t ≤ t0. We keep μ, M, S₁ separate to avoid any packing bugs.
    function h(p, s)
        μ = p.φ(s)
        M = μ * μ'
        # For now take S₁ history as zero; you can change this to e.g.
        # μ * p.φ(s - p.τ)' if you want a specific cross‑moment history.
        S1 = μ * p.φ(s - p.τ)'
        S2 = μ * p.φ(s - 2 * p.τ)'
        return ArrayPartition(μ, M, S1, S2)
    end

    u0 = h(p, first(tspan))

    time_history = [(t=0.0, dt=0.0)]
    
    function f!(du, u, h, p, t)
        dt = t - time_history[end].t;
        push!(time_history, (t=t, dt=dt))
        A, B, c, α, β, γ, τ = p.A, p.B, p.c, p.α, p.β, p.γ, p.τ
        μ, M, S1, S2 = u.x
        # Delayed state at t-τ
        μτ, Mτ, S1τ, S2τ = (h(p, t - τ)).x
        # For S₂ we need μ(t-2τ) and S₃(t) = E[x(t)x(t-3τ)']. We close S₃ with edge: S₃ = μ(t) μ(t-3τ)'
        μ₂τ = (h(p, t - 2 * τ)).x[1]
        μ₃τ = (h(p, t - 3 * τ)).x[1]
        S3_edge = μ * μ₃τ'   # S₃(t) when not in state

        # Mean
        dμ = A * μ .+ B * μτ .+ c

        # Second moment: dM = A*M + M*A' + B*S₁' + S₁*B' + c*μ' + μ*c' (+ diffusion if present)
        dM = A * M .+ M * A' .+ B * S1' .+ S1 * B' .+ c * μ' .+ μ * c'

        du.x[1] .= dμ
        du.x[2] .= dM

        # S₁(t) = E[x(t) x(t-τ)']: dS₁ = A*S₁ + S₁*A' + B*M(t-τ) + S₂*B' + c*μ(t-τ)' + μ*c'
        du.x[3] .= A * S1 + S1 * A' + B * Mτ + S2 * B' + c * μτ' + μ * c'

        # S₂(t) = E[x(t) x(t-2τ)']: dS₂ = A*S₂ + S₂*A' + B*S₁(t-τ) + S₃(t)*B' + c*μ(t-2τ)' + μ*c'
        # Skp1 must be S₃(t) (current-time), not S₂(t-τ). μtmkτ must be μ(t-2τ), not μ(t-τ).
        du.x[4] .= A * S2 + S2 * A' + B * S1τ + S3_edge * B' + c * μ₂τ' + μ * c'

        # for k in 1:5
        #     kidx = 2 + k;
        #     Skm1 = (h(p, t - τ)).x[kidx-1];
        #     Skp1 = k === 5 ? μ * (h(p, t-6τ)).x[1]' : u.x[kidx+1];
        #     μtmkτ = (h(p, t - k * τ)).x[1];
        #     du.x[kidx] .= A * u.x[kidx] + u.x[kidx] * A' + B * Skm1 + Skp1 * B' + c * μtmkτ' + μ * c'
        # end
        return nothing
    end

    # Declare all lags used in f!: τ, 2τ (for μ₂τ, S₁(t-τ)), 3τ (for S₃ edge closure)
    prob = DDEProblem(f!, u0, h, tspan, p; constant_lags=[τ0, 2*τ0, 3*τ0])
    sol = solve(prob; saveat=saveat, kwargs...)
    return sol
end

get_moments_manual(sol, t) = (sol(t).x...,)