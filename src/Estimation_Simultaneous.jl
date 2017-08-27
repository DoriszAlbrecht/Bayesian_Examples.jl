using DynamicHMC
using DiffWrappers
using ContinuousTransformations
include(Pkg.dir("Bayesian_Examples", "src", "Simultaneous.jl"))
using SimultaneousModel
using Distributions
import Distributions: Uniform
using JLD

RNG = Base.Random.GLOBAL_RNG

β = 0.9
C, Y = simulate_simultaneous(β, rand(Normal(100, 3), 100), rand(Normal(0, 5), 100))

# set up the model
model = Simultaneous(C, Y, Uniform(0, 1), Normal(100, 3), Normal(0, 5), 1000)

# we start the estimation process from the true values
θ₀ = inverse(model.transformation, β)


# wrap for gradient calculations

fgw = ForwardGradientWrapper(model, θ₀)

# sample

sample, tuned_sampler = NUTS_tune_and_mcmc(RNG, fgw, 5000; q = θ₀)

# posterior analysis

posterior = variable_matrix(sample)

sample_β = map_by_row(model.transformation, posterior)


# save

save(Simultaneous.path("results", "results.jld"), Dict("sample" => sample, "posterior" => posterior, "sample_β" => sample_β, "β" => β))
