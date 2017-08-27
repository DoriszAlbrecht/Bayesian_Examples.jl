module SimultaneousModel

using ArgCheck
using Distributions
using Parameters
using DynamicHMC
using StatsBase
using Base.Test
using ContinuousTransformations

export simulate_simultaneous, Simultaneous

############################################################################
## Simple simultaneous equations
############################################################################

## True model:
##  Cₜ = β * Yₜ + uₜ where  uₜ ∼ N(0, τ)   (1)
##  Yₜ = Cₜ + Xₜ                              (2)

## Cₜ is the consumption at time t
## Xₜis the non-consumption at time t
## Yₜ is the output at time t
## So output is used for consumption and non-consumption

## We have simultaneous equations as Cₜ depends on Yₜ
## And Yₜ depends on Cₜ as well


## Only Xₜ is exogenous in this model, Yₜ and Cₜ are endogenous

##############################################################################
## Functions needed for the model
##############################################################################

"""
    simulate_simultaneous(β, X, us)

Take in the parameter β, X and errors us, give back the endogenous variables of the system (Y and C).
"""
function simulate_simultaneous(β, X, us)
    N = length(us)
    C = Vector{eltype(X)}(N)
    Y = (X .+ us) / (1 - β)
    C = β * Y .+ us
    return (C, Y)
end


struct Simultaneous{T, Prior_β, Dist_x, Dist_us, Ttrans}
    "observed consumption"
    Cs::Vector{T}
    "observed output"
    Ys::Vector{T}
    "non-comsumption"
    Xs::Vector{T}
    "distribution of Xs"
    dist_x::Dist_x
    "prior for β"
    prior_β::Prior_β
    "Normal(0,τ) draws for simulation, where τ is fixed"
    us::Vector{T}
    "distribution of us"
    dist_us::Dist_us
    "transformation cached"
    transformation::Ttrans
end

ℝto(dist::Uniform) = transformation_to(Segment(minimum(dist), maximum(dist)), LOGISTIC)

"""
    Simultaneous(Cs, Ys, prior_β, dist_x, dist_us, M)

Convenience constructor for ToySimultaneousModel.
Take in the observed data, the prior, and number of simulations (M).
"""
function Simultaneous(Cs, Ys, prior_β, dist_x, dist_us, M)
    transformation = TransformationTuple((ℝto(prior_β)))
    Simultaneous(Cs, Ys, rand(dist_x, M), dist_x, prior_β, rand(dist_us, M), dist_us, transformation)
end



## logdensity uses the following auxiliary model:
## Cₜ ∼  N(β₁ + β₂ * Xₜ, √σ²)

function (pp::Simultaneous)(θ)
    @unpack Cs, Ys, Xs, prior_β, us, transformation = pp
    β = transformation(θ)
    logprior = logpdf(prior_β, β)
    Ones = ones(length(us))
    ## Generating the data
    C, Y = simulate_simultaneous(β, Xs, us)
    # OLS estimatation, regressing C on [1 X]
    XX = hcat(Ones, Y-C)
    est, σ² = qrfact(XX, Val{true}) \ C

    log_likelihood = sum(logpdf.(Normal(0, √σ²), Cs - [ones(length(Cs)) Ys-Cs] * est))
    return(logprior + log_likelihood)
end


"""
    path(parts...)

`parts...` appended to the directory containing `src/` etc. Useful for saving graphs and data.

"""

path(parts...) = normpath(joinpath(splitdir(Base.find_in_path("Simultaneous"))[1], "..", parts...))


end
