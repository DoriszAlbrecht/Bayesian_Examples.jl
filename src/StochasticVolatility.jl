module StochasticVolatility

using ArgCheck
using Distributions
using Parameters
using DynamicHMC
using StatsBase
using StatPlots
using Base.Test
using ForwardDiff
using Plots
using ContinuousTransformations
import ForwardDiff: Dual
using DiffWrappers
export StochasticVolatility
plotlyjs()


"""
    simulate_stochastic(ρ, σ_v, ϵs, νs)

Take in the parameter values (ρ, σ) for the latent volatility process, the errors ϵs used for the measurement equation and the errors νs used for the transition equation.

The discrete-time version of the Ornstein-Ulenbeck Stochastic - volatility model:

    y_t = x_t + ϵ_t where ϵ_t ∼ χ^2(1)
    x_t = ρ * x_(t-1) + σ * ν_t  where ν_t ∼ N(0, 1)

"""
function simulate_stochastic(ρ, σ, ϵs, νs)
    N = length(ϵs)
    @argcheck N == length(νs)
    x₀ = νs[1]*σ*(1 - ρ^2)^(-0.5)
    xs = Vector{typeof(x₀)}(N)
    for i in 1:N
        xs[i] = (i == 1) ? x₀ : (ρ*xs[i-1] + σ*νs[i])
    end
    xs + log.(ϵs) + 1.27
end

simulate_stochastic(ρ, σ, N) = simulate_stochastic(ρ, σ, rand(Chisq(1), N), randn(N))


struct StochasticVolatility{T, Prior_ρ, Prior_σ, Ttrans}
    "observed data"
    ys::Vector{T}
    "prior for ρ (persistence)"
    prior_ρ::Prior_ρ
    "prior for σ_v (volatility of volatility)"
    prior_σ::Prior_σ
    "χ^2 draws for simulation"
    ϵ::Vector{T}
    "Normal(0,1) draws for simulation"
    ν::Vector{T}
    "Transformations cached"
    transformation::Ttrans
end



## In this form, we use an AR(2) process of the first differences with an intercept as the auxiliary model.

"""
    lag(xs, n, K)

Lag-`n` operator on vector `xs` (maximum `K` lags).
"""
lag(xs, n, K) = xs[((K+1):end)-n]

"""
    lag_matrix(xs, ns, K = maximum(ns))

Matrix with differently lagged xs.
"""
function lag_matrix(xs, ns, K = maximum(ns))
    M = Matrix{eltype(xs)}(length(xs)-K, maximum(ns))
    for i ∈ ns
        M[:, i] = lag(xs, i, K)
    end
    M
end


"first auxiliary regression y, X, meant to capture first differences"
function yX1(zs, K)
    Δs = diff(zs)
    lag(Δs, 0, K), hcat(lag_matrix(Δs, 1:K, K), ones(eltype(zs), length(Δs)-K), lag(zs, 1, K+1))
end

"second auxiliary regression y, X, meant to capture levels"
function yX2(zs, K)
    lag(zs, 0, K), hcat(ones(eltype(zs), length(zs)-K), lag_matrix(zs, 1:K, K))
end

"Constructor which calculates cached values and buffers. Use this unless you know what you are doing."

bridge_trans(dist::Uniform) = bridge(ℝ, Segment(dist.a, dist.b))
bridge_trans(dist::InverseGamma) = bridge(ℝ, ℝ⁺)


"""
    Toy_Vol_Problem(ys, prior_ρ, prior_σ, M)

Convenience constructor for Toy_Vol_Problem.
Take in the observed data, the priors, and number of simulations (M).
"""
function StochasticVolatility(ys, prior_ρ, prior_σ, M)
    StochasticVolatility(ys, prior_ρ, prior_σ, rand(Chisq(1), M), randn(M))
end

function StochasticVolatility(ys, prior_ρ, prior_σ, ϵ, ν)
    transformation = TransformationTuple((bridge_trans(prior_ρ), (bridge_trans(prior_σ))))
    StochasticVolatility(ys, prior_ρ, prior_σ, ϵ, ν, transformation)
end


function (pp::StochasticVolatility)(θ)
    @unpack ys, prior_ρ, prior_σ, ν, ϵ, transformation = pp
    ρ, σ = transformation(θ)
    logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ, σ)
    N = length(ϵ)

    # Generating xs, which is the latent volatility process

    xs = simulate_stochastic(ρ, σ, ϵ, ν)
    Y_1, X_1 = yX1(xs, 2)
    β₁ = qrfact(X_1, Val{true}) \ Y_1
    v₁ = mean(abs2,  Y_1 - X_1*β₁)
    Y_2, X_2 = yX2(xs, 2)
    β₂ = qrfact(X_2, Val{true}) \ Y_2
    v₂ = mean(abs2,  Y_2 - X_2*β₂)
    # We work with first differences
    y₁, X₁ = yX1(ys, 2)
    log_likelihood1 = sum(logpdf.(Normal(0, √v₁), y₁ - X₁ * β₁))
    y₂, X₂ = yX2(ys, 2)
    log_likelihood2 = sum(logpdf.(Normal(0, √v₂), y₂ - X₂ * β₂))
    logprior + log_likelihood1 + log_likelihood2
end

"""

    path(parts...)

`parts...` appended to the directory containing `src/` etc. Useful for saving graphs and data. *NOT EXPORTED*.

"""

path(parts...) = normpath(joinpath(splitdir(Base.find_in_path("StochasticVolatility"))[1], "..", parts...))

end
