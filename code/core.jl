using DifferentialEquations
using LinearAlgebra

"""
Pack/unpack helpers:
State vector y contains: μ (n), M (n×n), S_1..S_K (each n×n) in that order.
"""
struct Layout
    n::Int
    K::Int
    idx_μ::UnitRange{Int}
    idx_M::UnitRange{Int}
    idx_S::Vector{UnitRange{Int}}  # length K
end

function make_layout(n, K)
    i = 1
    idx_μ = i:(i+n-1)
    i += n
    idx_M = i:(i+n*n-1)
    i += n * n
    idx_S = Vector{UnitRange{Int}}(undef, K)
    for k in 1:K
        idx_S[k] = i:(i+n*n-1)
        i += n * n
    end
    return Layout(n, K, idx_μ, idx_M, idx_S)
end

# reshape helpers
vecmat(M) = reshape(M, :)
unvecmat(v, n) = reshape(v, n, n)

# unpack current state
function unpack_state(y, L::Layout)
    n = L.n
    μ = @view y[L.idx_μ]
    M = unvecmat(@view(y[L.idx_M]), n)
    S = [unvecmat(@view(y[L.idx_S[k]]), n) for k in 1:L.K]
    return μ, M, S
end

# ---- Moment DDE RHS ----
# Parameters container
Base.@kwdef mutable struct MomentParams
    A::Matrix{Float64}
    B::Matrix{Float64}
    c::Vector{Float64}
    α::Matrix{Float64}        # multiplicative noise (scalar W) coefficient on x(t)
    β::Matrix{Float64}        # multiplicative noise coefficient on x(t-τ)
    γ::Vector{Float64}        # additive (affine) diffusion term
    τ::Float64
    K::Int
    φ::Function               # history function: φ(t)::Vector for t≤0
    layout::Layout
end

"""
Compute E[L Lᵀ] where L = α x + β x_τ + γ (scalar-Wiener case),
expanded in terms of (μ, M, C, N) with C = S₁, N = M_τ.
"""
function E_LLᵀ(α, β, γ, μ, μτ, M, C, N)
    # α M αᵀ + β N βᵀ + α C βᵀ + β Cᵀ αᵀ
    term_quad = α * M * α' + β * N * β' + α * C * β' + β * C' * α'
    # + α μ γᵀ + γ μᵀ αᵀ + β μτ γᵀ + γ μτᵀ βᵀ + γ γᵀ
    term_lin = α * μ * γ' + γ * μ' * α' + β * μτ * γ' + γ * μτ' * β' + γ * γ'
    return term_quad + term_lin
end

"""
RHS function for the big DDE of [μ; vec(M); vec(S₁); ...; vec(S_K)].
h(p, θ) returns the past state vector at t+θ (θ<0).
"""
function mom_rhs!(dy, y, h, p::MomentParams, t)
    A, B, c, α, β, γ, τ, K, L = p.A, p.B, p.c, p.α, p.β, p.γ, p.τ, p.K, p.layout
    n = L.n

    μ, M, S = unpack_state(y, L)

    # ← solver-supplied past at t-τ (uses history for ≤0, interpolated sol for >0)
    yτ = h(p, t - τ)
    μτ, Mτ, Sτ = unpack_state(yτ, L)

    # Edge term S_{K+1}(t): still history because t-(K+1)τ < 0 by construction
    ykp1 = h(p, t - (K + 1) * τ)        # packed moment state at negative time
    μ_edge, _, _ = unpack_state(ykp1, L)   # μ_edge == φ(t-(K+1)τ)
    S_Kp1 = μ * μ_edge'

    # C := S₁(t) if it exists
    C = (K >= 1) ? S[1] : zeros(n, n)

    # Mean
    dμ = A * μ + B * μτ + c

    # Second moment
    dM = A * M + M * A' + B * Mτ + C * B' + c * μ' + μ * c' +
         E_LLᵀ(α, β, γ, μ, μτ, M, C, Mτ)

    # Cross-moments
    for k in 1:K
        Sk = S[k]
        Skm1_delay = (k == 1) ? Mτ : Sτ[k-1]      # S_{k-1}(t-τ)

        Skp1 = (k < K) ? S[k+1] : S_Kp1           # S_{k+1}(t)

        # ← mean at t-kτ must come from solver's `h`, not from φ
        ykm = h(p, t - k * τ)
        μkm, _, _ = unpack_state(ykm, L)

        dSk = A * Sk + Sk * A' + B * Skm1_delay + Skp1 * B' + c * μkm' + μ * c'
        dy[L.idx_S[k]] .= vec(dSk)
    end

    dy[L.idx_μ] .= dμ
    dy[L.idx_M] .= vec(dM)
    return nothing
end


# ---- History (initial function) ----
"""
Build the history state y(t) for t≤0 from deterministic φ(t):
μ(t)=φ(t), M(t)=φφᵀ, S_k(t)=φ(t)φ(t-kτ)ᵀ for k=1..K.
"""
function history_state(p::MomentParams, t)
    n, K, τ, L = size(p.A, 1), p.K, p.τ, p.layout
    μ = p.φ(t)
    M = μ * μ'                        # deterministic history ⇒ no uncertainty
    S = [μ * (p.φ(t - k * τ))' for k in 1:K]
    y = zeros(length(L.idx_μ) + length(L.idx_M) + (K) * n * n)
    y[L.idx_μ] .= μ
    y[L.idx_M] .= vecmat(M)
    for k in 1:K
        y[L.idx_S[k]] .= vecmat(S[k])
    end
    return y
end

# ---- Driver API ----
"""
solve_moments(A,B,c,α,β,γ; τ, T, φ, tspan=(0.0,T), saveat=nothing)

Returns (sol, layout) where sol(t) gives the packed state; use `unpack_state(sol(t), layout)`.
"""
function solve_moments(A, B, c, α, β, γ; τ, T, φ, tspan=(0.0, T), saveat=nothing)
    n = size(A, 1)
    @assert size(A) == (n, n)
    @assert size(B) == (n, n)
    @assert size(α) == (n, n)
    @assert size(β) == (n, n)
    @assert length(c) == n
    @assert length(γ) == n

    # choose K = floor(T/τ) so that (K+1)τ > T (closure with φ on [0,T])
    K = Int(floor(T / τ))
    L = make_layout(n, K)

    p = MomentParams(A=A, B=B, c=c, α=α, β=β, γ=γ, τ=τ, K=K, φ=φ, layout=L)

    # initial state at t=0 from history
    y0 = history_state(p, 0.0)

    # history function wrapper
    # h!(out, p_, t) = (out .= history_state(p_, t))
    # h = (p_, θ) -> history_state(p_, θ)  #DifferentialEquations will call with θ≤0
    # Only returns the prescribed history for s ≤ 0
    h_history(p_, s) = history_state(p_, s)   # NO shadowing of `h` here

    dde = DDEProblem(mom_rhs!, h_history, tspan, p; constant_lags=[τ])

    sol = solve(dde; saveat=saveat)
    return sol, L
end

# ---- Convenience: read moments from solution ----
"""
get_moments_at(sol, layout, t) -> (μ, M, S::Vector{Matrix})
"""
function get_moments_at(sol, L::Layout, t)
    y = sol(t)
    return unpack_state(y, L)
end
