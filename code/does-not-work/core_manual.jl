using DifferentialEquations
using RecursiveArrayTools
using LinearAlgebra
using DelayDiffEq

alg = MethodOfSteps(Tsit5());

"""
Manual moment system with explicit state components (no layout packing).
State is `u = (μ::Vector, M::Matrix, S₁::Matrix, S₂::Matrix)` in an `ArrayPartition`.
S₁ = E[x(t)x(t-τ)'], S₂ = E[x(t)x(t-2τ)']. S₃ is closed as μ(t)μ(t-3τ)'.

Now includes stochastic terms for:
dx = (A x + B x(t-τ) + c) dt + (α x + β x(t-τ) + γ) dW
"""
function solve_moments_manual(A, B, c, α, β, γ; τ, T, φ,
                              τmax=τ, tspan=(0.0, T), saveat=nothing, kwargs...)
    n = size(A, 1)
    τ0 = τmax

    p = (A=A, B=B, c=c, α=α, β=β, γ=γ, τ=τ0, φ=φ)

    # History function for t ≤ t0
    function h(p, s)
        μ = p.φ(s)
        M = μ * μ'
        S1 = μ * p.φ(s - p.τ)'
        S2 = μ * p.φ(s - 2p.τ)'
        return ArrayPartition(μ, M, S1, S2)
    end

    u0 = h(p, first(tspan))

    # Helper: E[g g'] for g = α x0 + β x1 + γ,
    # given μ0=E[x0], μ1=E[x1], M00=E[x0 x0'], M11=E[x1 x1'], M01=E[x0 x1'].
    function EgEgT(α, β, γ, μ0, μ1, M00, M11, M01)
        # Terms: α M00 α' + α M01 β' + β M01' α' + β M11 β'
        G = α * M00 * α' +
            α * M01 * β' +
            β * M01' * α' +
            β * M11 * β'

        # Mean-affine cross terms + constant
        G .+= α * μ0 * γ' + γ * μ0' * α'
        G .+= β * μ1 * γ' + γ * μ1' * β'
        G .+= γ * γ'
        return G
    end

    function f!(du, u, hfun, p, t)
        A, B, c, α, β, γ, τ = p.A, p.B, p.c, p.α, p.β, p.γ, p.τ
        μ, M, S1, S2 = u.x
        M = (M + M')/2;

        # Values at delays
        μτ, Mτ, S1τ, S2τ = (hfun(p, t - τ)).x    
        Mτ = (Mτ + Mτ')/2;
        μ2τ, M2τ, S1_2τ, S2_2τ = (hfun(p, t - 2τ)).x       # at t-2τ
        M2τ = (M2τ + M2τ')/2;
        μ3τ, M3τ, S1_3τ, S2_3τ = (hfun(p, t - 3τ)).x       # at t-3τ
        M3τ = (M3τ + M3τ')/2;

        # S3 closure at current time: S3(t) ≈ μ(t) μ(t-3τ)'
        S3_edge = μ * μ3τ'

        # Mean
        dμ = A * μ .+ B * μτ .+ c
        du.x[1] .= dμ

        # ===== Second moment M(t) = E[x x'] =====
        # Drift part:
        dM = A * M .+ M * A' .+ B * S1' .+ S1 * B' .+ c * μ' .+ μ * c'

        # Diffusion part: E[g(t) g(t)']
        # Here x0 = x(t), x1 = x(t-τ)
        dM .+= EgEgT(α, β, γ, μ, μτ, M, Mτ, S1)

        du.x[2] .= dM

        # ===== Cross moment S1(t) = E[x(t) x(t-τ)'] =====
        # Drift part:
        dS1 = A * S1 .+ S1 * A' .+ B * Mτ .+ S2 * B' .+ c * μτ' .+ μ * c'

        # Diffusion overlap term: E[g(t-τ) g(t-τ)']
        # g(t-τ) depends on x(t-τ) and x(t-2τ)
        # Here x0 = x(t-τ), x1 = x(t-2τ), M01 = E[x(t-τ) x(t-2τ)'] = S1(t-τ)
        dS1 .+= EgEgT(α, β, γ, μτ, μ2τ, Mτ, M2τ, S1τ)

        du.x[3] .= dS1

        # ===== Cross moment S2(t) = E[x(t) x(t-2τ)'] =====
        # Drift part:
        # Uses S1(t-τ) = E[x(t-τ)x(t-2τ)'] and S3(t) closure
        dS2 = A * S2 .+ S2 * A' .+ B * S1τ .+ S3_edge * B' .+ c * μ2τ' .+ μ * c'

        # Diffusion overlap term: E[g(t-2τ) g(t-2τ)']
        # g(t-2τ) depends on x(t-2τ) and x(t-3τ)
        # M01 = E[x(t-2τ)x(t-3τ)'] = S1(t-2τ)
        dS2 .+= EgEgT(α, β, γ, μ2τ, μ3τ, M2τ, M3τ, S1_2τ)

        du.x[4] .= dS2

        return nothing
    end

    # Declare lags used: τ, 2τ, 3τ (for μ3τ, M3τ, and S3 edge)
    prob = DDEProblem(f!, u0, h, tspan, p; constant_lags=[τ0, 2τ0, 3τ0])
    sol = solve(prob, alg;reltol=1e-12, abstol=1e-12, d_discontinuities = [τ, 2τ, 3τ], saveat=saveat, kwargs...)
    return sol
end

get_moments_manual(sol, t) = (sol(t).x...,)

using DifferentialEquations
using RecursiveArrayTools
using LinearAlgebra

using DifferentialEquations
using RecursiveArrayTools
using LinearAlgebra

"""
Centered-moment system of depth K for:
dx = (A x + B x(t-τ) + c) dt + (α x + β x(t-τ) + γ) dW

State is ArrayPartition(μ, V, C1, C2, ..., Cdepth), where
  μ(t)  = E[x(t)]
  V(t)  = Cov(x(t), x(t))
  Ck(t) = Cov(x(t), x(t-kτ)) for k=1..depth

Closure: C_{depth+1}(t) ≡ 0

History (default): deterministic => V=Ck=0 for s≤t0, μ(s)=φ(s).

Notes:
- V is symmetric (we enforce symmetry in the derivative).
- Ck are generally NOT symmetric (do not symmetrize them).
- Requires constant_lags up to (depth+1)τ because diffusion terms use t-(k+1)τ.
"""
function solve_centered_moments(A, B, c, α, β, γ;
                                      τ, T, φ, depth::Int=2,
                                      tspan=(0.0, T), saveat=nothing, kwargs...)
    @assert depth ≥ 1 "depth must be ≥ 1"
    n = size(A, 1)
    T0 = first(tspan)

    p = (A=A, B=B, c=c, α=α, β=β, γ=γ, τ=τ, φ=φ, depth=depth)

    # History: μ = φ(s), and centered covariances = 0 (deterministic history)
    function h(p, s)
        μ = p.φ(s)
        V = zeros(eltype(μ), n, n)
        Cs = [zeros(eltype(μ), n, n) for _ in 1:p.depth]
        return ArrayPartition(μ, V, Cs...)
    end

    u0 = h(p, T0)

    # E[g g'] for g = α x0 + β x1 + γ with (x0, x1) = (x(s), x(s-τ))
    # computed from centered stats:
    #   E[x0x0']=V00+μ0μ0', E[x1x1']=V11+μ1μ1', E[x0x1']=C01+μ0μ1'
    function EgEgT(α, β, γ, μ0, μ1, V00, V11, C01)
        Ex0x0 = V00 .+ μ0 * μ0'
        Ex1x1 = V11 .+ μ1 * μ1'
        Ex0x1 = C01 .+ μ0 * μ1'
        Ex1x0 = Ex0x1'

        G = α * Ex0x0 * α' +
            α * Ex0x1 * β' +
            β * Ex1x0 * α' +
            β * Ex1x1 * β'

        # affine mean-noise cross terms + additive noise
        G .+= α * μ0 * γ' + γ * μ0' * α'
        G .+= β * μ1 * γ' + γ * μ1' * β'
        G .+= γ * γ'
        return G
    end

    function f!(du, u, hfun, p, t)
        A, B, c, α, β, γ, τ, depth = p.A, p.B, p.c, p.α, p.β, p.γ, p.τ, p.depth

        μ = u.x[1]
        V = u.x[2]
        Cs = u.x[3:end]  # Cs[k] == C_{k}(t) with k=1..depth

        # Precompute delayed states at t - jτ for j=0..depth+1
        # Each state contains (μj, Vj, C1j..Cdepth_j)
        μj = Vector{typeof(μ)}(undef, depth+2)
        Vj = Vector{typeof(V)}(undef, depth+2)
        Csj = Vector{Vector{typeof(V)}}(undef, depth+2)

        for j in 0:(depth+1)
            uj = hfun(p, t - j*τ).x
            μj[j+1] = uj[1]
            Vj[j+1] = uj[2]
            # Convert the trailing tuple (C1, C2, ..., Cdepth) into a vector
            # so it matches the declared element type of `Csj`.
            Csj[j+1] = collect(uj[3:end])
        end

        # Convenience: C1 at current time (needed in V diffusion term)
        C1 = Cs[1]

        # ---- Mean ----
        du.x[1] .= A * μ .+ B * μj[2] .+ c

        # ---- Variance V ----
        # V' = A V + V A' + B C1' + C1 B' + E[g(t)g(t)']
        du.x[2] .= A * V .+ V * A' .+ B * C1' .+ C1 * B'
        du.x[2] .+= EgEgT(α, β, γ, μj[1], μj[2], Vj[1], Vj[2], Csj[1][1])

        # Enforce symmetry of dV (V must remain symmetric)
        du.x[2] .= (du.x[2] .+ du.x[2]') ./ 2

        # ---- Cross-covariances Ck ----
        # For k=1:
        #   C1' = A C1 + C1 A' + B V(t-τ) + C2 B' + E[g(t-τ)g(t-τ)']
        # For k>=2:
        #   Ck' = A Ck + Ck A' + B C_{k-1}(t-τ) + C_{k+1}(t) B' + E[g(t-kτ)g(t-kτ)']
        # Closure: C_{depth+1}(t) = 0
        for k in 1:depth
            Ck = Cs[k]

            # Drift coupling terms
            if k == 1
                D = Vj[2]                  # V(t-τ)
            else
                D = Csj[2][k-1]            # C_{k-1}(t-τ)
            end
            E = (k < depth) ? Cs[k+1] : zeros(eltype(V), n, n)  # C_{k+1}(t) or closure

            dCk = A * Ck .+ Ck * A' .+ B * D .+ E * B'

            # Diffusion overlap term at s = t - kτ:
            # add E[g(s)g(s)'] with (x(s), x(s-τ)) = (x(t-kτ), x(t-(k+1)τ))
            # needs C1(s) = Cov(x(s), x(s-τ)) which is C1 at time (t-kτ)
            C1_at_s = Csj[k+1][1]  # C1(t-kτ)
            dCk .+= EgEgT(α, β, γ,
                          μj[k+1], μj[k+2],
                          Vj[k+1], Vj[k+2],
                          C1_at_s)

            du.x[2+k] .= dCk
        end

        return nothing
    end

    lags = [j*τ for j in 1:(depth+1)]
    prob = DDEProblem(f!, u0, h, tspan, p; constant_lags=lags)
    return solve(prob, alg; reltol=1e-12, abstol=1e-12, adaptive=false, dt=τ/2000, tstops = lags, saveat=saveat, kwargs...)
end

# Access helper: returns (μ, V, C1, ..., Cdepth)
get_centered_moments(sol, t) = (sol(t).x...,)

using DifferentialEquations
using RecursiveArrayTools
using LinearAlgebra

"""
Centered-moment method with delay-line discretization.

SDDE:
  dx = (A x + B x(t-τ) + c) dt + (α x + β x(t-τ) + γ) dW

Approximation:
  Use substep h = τ/m.
  Track C[j](t) = Cov(x(t), x(t-jh)) for j=1..J where J = m*depth.

Special delay τ corresponds to index m (since τ = m*h).

State: ArrayPartition(μ, V, C1, C2, ..., CJ)

Closure: C[J+1](t) = 0

History default: deterministic => V=0 and all C[j]=0 for t≤t0, μ(s)=φ(s).
"""
function solve_centered_moments_delayline(A, B, c, α, β, γ;
                                         τ, T, φ,
                                         m::Int=50,
                                         depth::Int=1,
                                         tspan=(0.0, T),
                                         saveat=nothing,
                                         kwargs...)
    @assert m ≥ 1
    @assert depth ≥ 1
    n = size(A, 1)

    h = τ / m
    J = m * depth

    p = (A=A, B=B, c=c, α=α, β=β, γ=γ, τ=τ, h=h, m=m, J=J, φ=φ)

    # History: deterministic centered moments
    function hist(p, s)
        μ = p.φ(s)
        V = zeros(eltype(μ), n, n)
        Cs = [zeros(eltype(μ), n, n) for _ in 1:p.J]
        return ArrayPartition(μ, V, Cs...)
    end

    u0 = hist(p, first(tspan))

    # Compute E[g g'] for g = α x0 + β xτ + γ, where xτ is x(t-τ)
    # Here: x0 = x(t), xτ = x(t-m*h)
    # Inputs are centered moments:
    #   E[x0x0'] = V00 + μ0μ0'
    #   E[xτxτ'] = Vττ + μτμτ'
    #   E[x0xτ'] = C0τ + μ0μτ'
    function EgEgT(α, β, γ, μ0, μτ, V00, Vττ, C0τ)
        Ex0x0 = V00 .+ μ0 * μ0'
        Exτxτ = Vττ .+ μτ * μτ'
        Ex0xτ = C0τ .+ μ0 * μτ'
        Exτx0 = Ex0xτ'

        G = α * Ex0x0 * α' +
            α * Ex0xτ * β' +
            β * Exτx0 * α' +
            β * Exτxτ * β'

        G .+= α * μ0 * γ' + γ * μ0' * α'
        G .+= β * μτ * γ' + γ * μτ' * β'
        G .+= γ * γ'
        return G
    end

    function f!(du, u, hfun, p, t)
        A, B, c, α, β, γ = p.A, p.B, p.c, p.α, p.β, p.γ
        h, m, J = p.h, p.m, p.J

        μ = u.x[1]
        V = u.x[2]
        Cs = u.x[3:end]  # Cs[j] = C[j] = Cov(x(t), x(t-jh)), j=1..J

        # Values at t - jh for j = 0..J+m.
        # Each contains μj, Vj, C1..CJ at that time.
        # We need up to j+m because diffusion terms for C[J] require
        # access to statistics at time t-(J+m)h.
        μj = Vector{typeof(μ)}(undef, J + m + 1)
        Vj = Vector{typeof(V)}(undef, J + m + 1)
        Csj = Vector{Vector{typeof(V)}}(undef, J + m + 1)

        for j in 0:(J + m)
            uj = hfun(p, t - j*h).x
            μj[j+1] = uj[1]
            Vj[j+1] = uj[2]
            Csj[j+1] = collect(uj[3:end])
        end

        # convenience: at current time
        C_m = Cs[m]  # Cov(x(t), x(t-τ))

        # ---- Mean ----
        du.x[1] .= A * μ .+ B * μj[m+1] .+ c

        # ---- Variance V ----
        # V' = A V + V A' + B C_m' + C_m B' + E[g(t)g(t)']
        du.x[2] .= A * V .+ V * A' .+ B * C_m' .+ C_m * B'
        du.x[2] .+= EgEgT(α, β, γ,
                          μj[1], μj[m+1],
                          Vj[1], Vj[m+1],
                          Cs[m])

        du.x[2] .= (du.x[2] .+ du.x[2]') ./ 2  # enforce symmetry for V

        # ---- Cross-covariances C[j] ----
        #
        # For general lag ℓ = jh:
        # d/dt Cov(x(t), x(t-ℓ)) =
        #   A C[j] + C[j] A'
        # + B Cov(x(t-τ), x(t-ℓ)) + Cov(x(t), x(t-ℓ-τ)) B'
        # + E[g(t-ℓ) g(t-ℓ)']
        #
        # Here:
        #   Cov(x(t-τ), x(t-ℓ)) = Cov(x(t-mh), x(t-jh))
        #     which is "backward" cross-cov at time t-jh with lag (m-j)h if m>j,
        #     or at time t-mh with lag (j-m)h if j>m.
        #
        # But we only store forward covariances from "current time" at each history point.
        # We can retrieve needed terms using identities:
        #   Cov(x(a), x(b)) = Cov(x(b), x(a))'
        #
        # We'll compute the two drift-cov terms as:
        #   Term1 = Cov(x(t-τ), x(t-jh)) = Cov(x(t-jh), x(t-τ))'
        #         = C_at_(t-jh)[|m-j|] with transpose depending on ordering.
        #   Term2 = Cov(x(t), x(t-jh-τ)) = C[j+m] if j+m ≤ J else closure 0.
        #
        for j in 1:J
            Cj = Cs[j]

            # Term2: Cov(x(t), x(t-(j+m)h)) = C[j+m] (or 0 by closure)
            Term2 = (j + m <= J) ? Cs[j+m] : zeros(eltype(V), n, n)

            # Term1: Cov(x(t-τ), x(t-jh)) = Cov(x(t-mh), x(t-jh))
            # Use whichever time is later as "current" in stored C's.
            if j == m
                Term1 = Vj[m+1]  # Cov(x(t-mh), x(t-mh)) = V(t-τ)
            elseif j < m
                # later time is t-jh, earlier is t-mh
                # Cov(x(t-jh), x(t-mh)) is stored at time t-jh as C[m-j]
                Term1 = Csj[j+1][m-j]'  # transpose to flip order
            else # j > m
                # later time is t-mh, earlier is t-jh
                # Cov(x(t-mh), x(t-jh)) stored at time t-mh as C[j-m]
                Term1 = Csj[m+1][j-m]   # already in correct order
            end

            # Drift part
            dC = A * Cj .+ Cj * A' .+ B * Term1 .+ Term2 * B'

            # Diffusion overlap term at s = t - jh:
            # Add E[g(s)g(s)'] where g(s)=αx(s)+βx(s-τ)+γ and τ=mh.
            # Here x(s)=x(t-jh), x(s-τ)=x(t-(j+m)h).
            μ_s = μj[j+1]
            V_s = Vj[j+1]
            μ_sτ = μj[j+m+1]
            V_sτ = Vj[j+m+1]
            # C0τ at time s is Cov(x(s), x(s-τ)) = C[m] evaluated at time s = t-jh
            C_s_m = Csj[j+1][m]

            dC .+= EgEgT(α, β, γ, μ_s, μ_sτ, V_s, V_sτ, C_s_m)

            du.x[2+j] .= dC
        end

        return nothing
    end

    maxLagSteps = J + m
    lags = [k*h for k in 1:maxLagSteps]

    prob = DDEProblem(f!, u0, hist, tspan, p; constant_lags=lags)
    return solve(prob; saveat=saveat, kwargs...)
end

using DifferentialEquations
using LinearAlgebra

# Helpers to index blocks
@inline function blockrange(n::Int, j::Int)
    # j = 0..m
    (j*n + 1):((j+1)*n)
end

"""
Build extended drift matrix A_e and drift vector c_e for the upwind transport delay line.

z = [x0; x1; ...; xm], each xj ∈ R^n, N = n*(m+1)
x0' = A x0 + B xm + c
xj' = (1/h)(x_{j-1} - xj), j=1..m
"""
function build_extended_drift(A, B, c, τ; m::Int)
    n = size(A,1)
    @assert size(A,2)==n
    @assert size(B,1)==n && size(B,2)==n
    @assert length(c)==n
    @assert m ≥ 1

    h = τ/m
    N = n*(m+1)

    Ae = zeros(eltype(A), N, N)
    ce = zeros(eltype(c), N)

    # x0 block
    r0 = blockrange(n, 0)
    Ae[r0, r0] .= A
    Ae[r0, blockrange(n, m)] .+= B
    ce[r0] .= c

    # shift chain blocks
    I_n = Matrix{eltype(A)}(I, n, n)
    for j in 1:m
        rj = blockrange(n, j)
        rjm1 = blockrange(n, j-1)
        rjp1 = blockrange(n, j+1)
        if j<m
            Ae[rj, rjm1] .+= (1/(2h)) .* I_n
            Ae[rj, rjp1]   .+= (-1/(2h)) .* I_n
        else
            Ae[rj, rj]   .+= (-1/h) .* I_n
            Ae[rj, rjm1] .+= (1/h) .* I_n
        end
    end

    return Ae, ce, h
end

"""
Compute E[g g'] where g = α x0 + β xm + γ
using μ and covariance P of the extended state.
"""
function EgEgT_from_muP(α, β, γ, μ0, μm, P00, Pmm, P0m)
    # raw second moments
    Ex0x0 = P00 .+ μ0*μ0'
    Exmxm = Pmm .+ μm*μm'
    Ex0xm = P0m .+ μ0*μm'
    Exmx0 = Ex0xm'

    G = α*Ex0x0*α' + α*Ex0xm*β' + β*Exmx0*α' + β*Exmxm*β'
    G .+= α*μ0*γ' + γ*μ0'*α'
    G .+= β*μm*γ' + γ*μm'*β'
    G .+= γ*γ'
    return G
end

"""
Solve mean/covariance ODE for the extended-state approximation.

Returns:
  sol: ODESolution over state y = [vec(μ); vec(P)]
Extraction:
  N = n*(m+1)
  μ(t) = reshape(sol(t)[1:N], N)
  P(t) = reshape(sol(t)[N+1:end], N, N)

Mean/variance of original x(t):
  μx(t) = μ0 block (j=0)
  Vx(t) = P00 block (j=0,j=0)
"""
function solve_extended_moments(A, B, c, α, β, γ; τ, T, φ, m::Int=200,
                                tspan=(0.0,T), kwargs...)
    n = size(A,1)
    Ae, ce, h = build_extended_drift(A, B, c, τ; m=m)
    N = n*(m+1)

    # Initial mean μ(0): fill delay line with history φ on [-τ,0]
    # xj(0) ≈ φ(-j*h)
    μ0 = zeros(eltype(c), N)
    for j in 0:m
        μ0[blockrange(n,j)] .= φ(-j*h)
    end

    # Initial covariance P(0): typically zero if deterministic history
    P0 = zeros(eltype(c), N, N)

    y0 = vcat(vec(μ0), vec(P0))

    # Preallocate workspaces for the ODE RHS to avoid per-step allocations
    r0 = blockrange(n, 0)
    rm = blockrange(n, m)
    AeT = transpose(Ae)
    tmpP1 = similar(P0)
    tmpP2 = similar(P0)
    Q = similar(P0)
    zeroP = zero(eltype(P0))

    function f!(dy, y, hfun, p, t)
    # function f!(dy, y, p, t)
        @views μ = y[1:N]
        @views P = reshape(y[N+1:end], N, N)

        @views dμ = dy[1:N]
        @views dP = reshape(dy[N+1:end], N, N)

        # dμ = Ae * μ + ce (in-place)
        mul!(dμ, Ae, μ)
        dμ .+= ce

        # pull blocks for diffusion term (only affects x0 block)
        @views μ_0 = μ[r0]
        @views μ_m = μ[rm]

        @views P00 = P[r0, r0]
        @views Pmm = P[rm, rm]
        @views P0m = P[r0, rm]

        EgEg = EgEgT_from_muP(α, β, γ, μ_0, μ_m, P00, Pmm, P0m)

        # Q only has nonzero entries in the (0,0) block; reuse workspace
        fill!(Q, zeroP)
        @views Q[r0, r0] .= EgEg

        # dP = Ae*P + P*Ae' + Q, computed with workspaces
        mul!(tmpP1, Ae, P)
        mul!(tmpP2, P, AeT)
        dP .= tmpP1
        dP .+= tmpP2
        dP .+= Q

        return nothing
    end

    hist(p, s) = y0;

    prob = DDEProblem(f!, y0, hist, tspan)
    # prob = ODEProblem(f!, y0, tspan)
    sol = solve(prob; adaptive=false, dt=τ/1000, kwargs...)
    return sol, (Ae=Ae, ce=ce, h=h, N=N, n=n, m=m)
end

"""
Helper to extract μx(t), Vx(t) from solution.
"""
function get_x_moments(sol, meta, t)
    N,n,m = meta.N, meta.n, meta.m
    y = sol(t)
    μ = reshape(view(y, 1:N), N)
    P = reshape(view(y, N+1:length(y)), N, N)

    r0 = blockrange(n,0)
    μx = μ[r0]
    Vx = P[r0, r0]
    return μx, Vx
end