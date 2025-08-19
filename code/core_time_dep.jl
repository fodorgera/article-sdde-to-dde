using DifferentialEquations
using LinearAlgebra

"State layout (unchanged)"
struct Layout
    n::Int
    K::Int
    idx_μ::UnitRange{Int}
    idx_M::UnitRange{Int}
    idx_S::Vector{UnitRange{Int}}
end

function make_layout(n, K)
    println("K: ", K)
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
    Layout(n, K, idx_μ, idx_M, idx_S)
end

vecmat(M) = reshape(M, :)
unvecmat(v, n) = reshape(v, n, n)

function unpack_state(y, L::Layout)
    n = L.n
    μ = @view y[L.idx_μ]
    M = unvecmat(@view(y[L.idx_M]), n)
    S = [unvecmat(@view(y[L.idx_S[k]]), n) for k in 1:L.K]
    return μ, M, S
end

# --- NEW: helpers to allow constants or functions of time ---
mat_at(M, t) = M isa Function ? M(t) : M
vec_at(v, t) = v isa Function ? v(t) : v

# ---- Parameters ----
Base.@kwdef mutable struct MomentParams
    A::Any            # Matrix{Float64} OR (t->Matrix)
    B::Any            # "
    c::Any            # Vector{Float64} OR (t->Vector)
    α::Any            # Matrix OR (t->Matrix)
    β::Any            # "
    γ::Any            # Vector OR (t->Vector)
    τ::Float64
    K::Int
    φ::Function       # history x(t) for t≤0 → Vector
    layout::Layout
end

"Compute E[L Lᵀ] with L = α x + β x_τ + γ, given moments at time t."
function E_LLᵀ(αt, βt, γt, μ, μτ, M, C, N)
    term_quad = αt * M * αt' + βt * N * βt' + αt * C * βt' + βt * C' * αt'
    term_lin = αt * μ * γt' + γt * μ' * αt' + βt * μτ * γt' + γt * μτ' * βt' + γt * γt'
    term_quad + term_lin
end

"""
RHS for y = [μ; vec(M); vec(S₁); …; vec(S_K)].
h(p, s) returns state at absolute time s (as in DiffEq docs).
"""
function mom_rhs!(dy, y, h, p::MomentParams, t)
    n, K, L = p.layout.n, p.K, p.layout

    # coefficients at current time t
    At = mat_at(p.A, t)
    Bt = mat_at(p.B, t)
    αt = mat_at(p.α, t)
    βt = mat_at(p.β, t)
    ct = vec_at(p.c, t)
    γt = vec_at(p.γ, t)

    μ, M, S = unpack_state(y, L)

    # delayed state at t-τ (absolute time API)
    yτ = h(p, t - p.τ)
    μτ, Mτ, Sτ = unpack_state(yτ, L)

    # edge term S_{K+1}(t): relies on history since t-(K+1)τ < 0
    y_edge = h(p, t - (K + 1) * p.τ)
    μ_edge, _, _ = unpack_state(y_edge, L)           # equals φ(...) for deterministic history
    S_Kp1 = μ * μ_edge'

    # S₁ alias
    C = (K >= 1) ? S[1] : zeros(n, n)

    # mean
    dμ = At * μ + Bt * μτ + ct

    # second moment
    dM = At * M + M * At' + Bt * Mτ + C * Bt' + ct * μ' + μ * ct' +
         E_LLᵀ(αt, βt, γt, μ, μτ, M, C, Mτ)

    # cross-moments S_k
    for k in 1:K
        Sk = S[k]
        Skm1_delay = (k == 1) ? Mτ : Sτ[k-1]         # S_{k-1}(t-τ)
        Skp1 = (k < K) ? S[k+1] : S_Kp1   

        # mean at t - kτ (absolute time)
        ykm = h(p, t - k * p.τ)
        μkm, _, _ = unpack_state(ykm, L)

        dSk = At * Sk + Sk * At' + Bt * Skm1_delay + Skp1 * Bt' + ct * μkm' + μ * ct'
        dy[L.idx_S[k]] .= vec(dSk)
    end

    dy[L.idx_μ] .= dμ
    dy[L.idx_M] .= vec(dM)
    return nothing
end

"History → build y(t) for t≤0 from φ(t) (deterministic history)."
function history_state(p::MomentParams, t)
    n, K, τ, L = size(mat_at(p.A, 0.0), 1), p.K, p.τ, p.layout
    μ = p.φ(t)
    M = μ * μ'
    S = [μ * (p.φ(t - k * τ))' for k in 1:K]
    y = zeros(length(L.idx_μ) + length(L.idx_M) + K * n * n)
    y[L.idx_μ] .= μ
    y[L.idx_M] .= vec(M)
    for k in 1:K
        y[L.idx_S[k]] .= vec(S[k])
    end
    y
end

"""
Driver.
A,B,α,β may be Matrix or t->Matrix; c,γ may be Vector or t->Vector.
"""
function solve_moments(A, B, c, α, β, γ; τ, T, φ, tspan=(0.0, T), saveat=nothing, depth=1)
    # infer n from A at t=0
    A0 = mat_at(A, 0.0)
    n = size(A0, 1)
    @assert size(A0) == (n, n)

    # K = Int(floor(T / τ))
    K = depth;
    L = make_layout(n, K)

    p = MomentParams(A=A, B=B, c=c, α=α, β=β, γ=γ, τ=τ, K=K, φ=φ, layout=L)

    y0 = history_state(p, 0.0)

    h_history(p_, s) = history_state(p_, s)  # used only for s ≤ 0
    dde = DDEProblem(mom_rhs!, h_history, tspan, p; constant_lags=[τ])
    sol = solve(dde; saveat=saveat)
    return sol, L
end

"Helper to extract at time t"
get_moments_at(sol, L::Layout, t) = unpack_state(sol(t), L)