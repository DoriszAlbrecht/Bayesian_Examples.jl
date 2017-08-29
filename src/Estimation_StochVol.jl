using DynamicHMC
using DiffWrappers
using ContinuousTransformations
include(Pkg.dir("Bayesian_Examples", "src", "StochasticVolatility.jl"))
using StochasticVolatilities

import Distributions: Uniform, InverseGamma
using JLD

RNG = Base.Random.GLOBAL_RNG

ρ = 0.8
σ = 0.6
y = simulate_stochastic(ρ, σ, 10000)

# setting up the model
model = StochasticVolatility(y, Uniform(-1, 1), InverseGamma(1, 1), 10000)

# we start the estimation process from the true values
θ₀ = inverse(model.transformation, (ρ, σ))


# wrap for gradient calculations

fgw = ForwardGradientWrapper(model, θ₀)

# sample

sample, tuned_sampler = NUTS_tune_and_mcmc(RNG, fgw, 5000; q = θ₀)

# posterior analysis

posterior = variable_matrix(sample)

sample_ρ, sample_σ = map_by_row(model.transformation, posterior)


# save

mkpath(StochasticVolatilities.path("results")) # ensure directory exists
save(StochasticVolatilities.path("results", "stochvol_results.jld"),
     Dict("sample" => sample, "posterior" => posterior, "sample_ρ" => sample_ρ,
          "sample_σ" => sample_σ, "ρ" => ρ, "σ" => σ))
