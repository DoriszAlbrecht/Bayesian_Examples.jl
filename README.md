
# Bayesian_Examples.jl
Examples of my project for Google Summer of Code

# Literature Overview
  * Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
  * Drovandi, C. C., Pettitt, A. N., & Lee, A. (2015). Bayesian indirect inference using a parametric auxiliary model. 
  * Gallant, A. R. and McCulloch, R. E. (2009). On the Determination of General Scientific Models With Application to Asset Pricing
  * Rayner, G. D. and MacGillivray, H. L. (2002). Numerical maximum likelihood estimation for the g-and-k and generalized g-and-h distributions. Stat. Comput. 12 57–75.

# GSOC 2017 project: Bayesian estimation using Random Walk Metropolis-Hastings and Dynamic Hamiltonian Monte Carlo methods

This summer I have had the opportunity to participate in the Google Summer of Code program. My project was in the Julia language and the main goal was to  with the purpose to use different methods to estimate models in order to improve efficiency. 

Under the mentorship of Tamás K. Papp, I completed a major revision of Bayesian estimation methods using Indirect Inference (II) and Hamiltonian Monte Carlo. I also got familiar with using git, opening issues, creating a repository among others. 

Hopefully, by the end of this post, I will manage to introduce these methods a little bit better and more extensively.

# Parametric Bayesian Indirect Likelihood for the Full Data

Usually when we face an intractable likelihood or a likelihood that would be extremely costly to calculate, we have the option to use an alternative auxiliary model to extract and estimate the parameters of interest. These alternative models should be easier to deal with. Drovandi et al. revises a collection of parametric Bayesian Indirect Inference (pBII) methods, I focused on the parametric Bayesian Indirect Likelihood for the Full Data (pdBIL) method proposed by Gallant and McCulloch (2009). The pdBIL uses the likelihood of the auxiliary model as a substitute for the intractable likelihood. The pdBIL does not compare summary statistics, instead works in the following way: 

First the data is generated, once we have the data, we can estimate the parameters of the auxiliary model. Then, the estimated parameters are put into the auxiliary likelihood with the observed/generated data. Afterwards we can use this likelihood in our chosen Bayesian method i.e. MCMC. 

To summarize the method, first we have the parameter vector θ and the observed data y. We would like to calculate the likelihood of p(θ|y), but it is intractable or costly to compute. In this case, with pdBIL we have to find an auxiliary model (A) that we use to approximate the true likelihood in the following way: 
First we have to generate points, denote with **x** from the data generating function with the previously proposed parameters θ. Then we compute the MLE of the auxiliary likelihood under **x** to get the parameters denoted by ϕ. Under these parameters ϕ, we can now compute the likelihood of pₐ(y|ϕ). It is desirable to have the auxiliary likelihood as close to the true likelihood as possible. 

# First stage of my project

In the first stage of my project I coded two models from Drovandi et al. using pdBIL. After calculating the likelihood of the auxiliary model, I used a Random Walk Metropolis-Hastings MCMC to sample from the target distribution [Toy models](https://github.com/tpapp/HamiltonianABC.jl/tree/dorisz-toy-models). In this stage of the project, the methods I used were well-known.
The purpose of the replication of the toy models from Drovandi et al. was to find out what issues we might face later and to come up with a usable interface. 
This stage resulted in [HamiltonianABC](https://github.com/tpapp/HamiltonianABC.jl/) (collaboration with Tamás K. Papp).

# Second stage of my project

After the first stage, I revised Betancourt (2017) and did a code revision for Tamás K. Papp's [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) which consisted of checking the code and its comparison with the paper. In addition to using the Hamiltonian Monte Carlo method, the usage of the forward mode automatic differentiation of the ForwardDiff package was the other main factor of this stage.
The novelty of this project was to find a way to fit every component together in a way to get an efficient estimation out of it.

# Stochastic Volatility model


The discrete-time version of the Ornstein-Ulenbeck Stochastic - volatility model:

    yₜ = xₜ + ϵₜ where ϵₜ ∼ χ²(1)
    xₜ = ρ * xₜ₋₁ + σ * νₜ  where νₜ ∼ N(0, 1)

For the auxiliary model, we used two regressions. The first regression was an AR(2) process on the first differences, the second was also an AR(2) process on the original variables in order to capture the levels. I will go into more details with this model.


I will now introduce the required steps for the estimation of the parameters of interest in the stochastic volatility model with the Dynamic Hamiltonian Monte Carlo method. We need to import three functions from the DynamicHMC repository: _logdensity_, _loggradient_ and _length_. 

* First we need to define a structure, which we will use in every imported function. This structure should contain the observed data, the priors and the shocks, but the components may vary depending on the estimated model. 

```julia
struct Volatility_Problem{T, Prior_ρ, Prior_σ}
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
end
```
After specifying the data generating function and a couple of facilitator and additional functions for the particular model, we can define the logdensity(pp::Structure\_of\_Model, θ) where the first term is the previously defined sturture, that is unpacked inside of the function, θ is the vector of parameters. 

```julia
function logdensity(pp::Volatility_Problem, θ)
    
    @unpack ys, prior_ρ, prior_σ, ν, ϵ = pp
    trans = transformation_to(pp)
    ρ, σ = trans(θ)
    logprior = logpdf(prior_ρ, ρ) + logpdf(prior_σ, σ)

    # Generating zs, which is the latent volatility process
    zs = simulate_stochastic(ρ, σ, ϵ, ν)
    β₁, v₁ = OLS(yX1(zs, 2)...)
    β₂, v₂ = OLS(yX2(zs, 2)...)
    
    # first differences
    y₁, X₁ = yX1(ys, 2)
    log_likelihood1 = sum(logpdf.(Normal(0, √v₁), y₁ - X₁ * β₁))

    # levels
    y₂, X₂ = yX2(ys, 2)
    log_likelihood2 = sum(logpdf.(Normal(0, √v₂), y₂ - X₂ * β₂))
    # likelihood
    logprior + log_likelihood1 + log_likelihood2 + logjac(trans, θ)

end
```
The next step is to define the loggradient(pp::Structure\_of\_Model, x) function. I used the ForwardDiff.jl package and its forward mode automatic differentiation method. The ForwardDiff.gradient gives back ∇logdensity evaluated at x. 

```julia
loggradient(pp::Volatility_Problem, x) =
    ForwardDiff.gradient(y->logdensity(pp, y), x)::Vector{Float64}
```
Finally, with the imported _length_ function, we have to specify the length of the parameters of interest. 
```julia
Base.length(::Volatility_Problem) = 2
```
Given the defined functions, we can now sample the parameters: 

```julia
const RNG = srand(UInt32[0x23ef614d, 0x8332e05c, 0x3c574111, 0x121aa2f4])
sample, tuned_sampler = NUTS_tune_and_mcmc(RNG, pp, 1000; q = θ₀)
```

The following graphs show the results for the parameters:

![rho_plot](https://user-images.githubusercontent.com/26724827/29598603-dd6d1ae2-8797-11e7-9837-5373c03c4ceb.png)


![sigma_plot](https://user-images.githubusercontent.com/26724827/29598635-0e84b9aa-8798-11e7-9218-bc62347407ae.png)

# Problems that I have faced during GSOC

1) Isolation.

  * The true model was the g-and-k quantile function described by Rayner and MacGillivray (2002). 
  * The auxiliary model was a three component normal mixture model. 

We faced serious problems with this model. \
First of all, I coded the MLE of the finite component normal mixture model, which computes the means, variances and weights of the normals given the observed data and the desired number of mixtures. 
With the g-and-k quantile function, I experienced the so called "isolation", which means that one observation point is an outlier getting weight 1, the other observed points get weigth 0, which results in variance equal to 0. There are ways to disentangle the problem of isolation, but the parameters of interests still did not converge to the true values. There is work to be done with this model.

2) Type-stability issues.
  
  To use the automatic differentiation method efficiently, I had to code the functions to be type-stable, otherwise the sampling functions would have taken hours to run. See the following example:
  
  * This is not type-stable
  ```julia
  function simulate_stochastic(ρ, σ, ϵs, νs)
    N = length(ϵs)
    @argcheck N == length(νs) 
    xs = Vector(N)
    for i in 1:N
        xs[i] = (i == 1) ? νs[1]*σ*(1 - ρ^2)^(-0.5) : (ρ*xs[i-1] + σ*νs[i])
    end
    xs + log.(ϵs) + 1.27
end
  ```
* This is type-stable
```julia
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
```



# Cut/Get rid of this chunk

**First model** 
  * The true model is y ∼ Exponential(λ), IID, where λ is the scale. 
  * The auxiliary model is y ∼ N(μ, σ²), with statistics ϕ = (μ, σ). 
  * The prior is λ ∼ Uniform(A,B) prior. 

This model and the estimation worked fine, but it is too simple to improve efficiency.

**Second model** 
  * The true model was the g-and-k quantile function described by Rayner and MacGillivray (2002). 
  * The auxiliary model was a three component normal mixture model. 

We faced serious problems with this model. \
First of all, I coded the MLE of the finite component normal mixture model, which computes the means, variances and weights of the normals given the observed data and the desired number of mixtures. 
With the g-and-k quantile function, I experienced the so called "isolation", which means that one observation point is an outlier getting weight 1, the other observed points get weigth 0, which results in variance equal to 0. There are ways to disentangle the problem of isolation, but the parameters of interests still did not converge to the true values. There is work to be done with this model.

Afterwards, we turned to two economic-related models:
* Stochastic volatility model
* Simultaneous equations


 
**Simultaneous equations** \
  True model:

  Cₜ = β * Yₜ + uₜ where  uₜ ∼ N(0, τ)   (1)\
  Yₜ = Cₜ + Xₜ                              (2)

  * Cₜ is the consumption at time t 
  * Xₜ is the non-consumption at time t
  * Yₜ is the output at time t 
  * Output is used for consumption and non-consumption 
  * We have simultaneous equations as Cₜ depends on Yₜ 
  * Also, Yₜ depends on Cₜ as well 
  * Only Xₜ is exogenous in this model, Yₜ and Cₜ are endogenous 

Auxiliary model: \
I assumed that the consumption at time t is normally distributed as follows \
       Cₜ ∼  N(β₁ + β₂ * Xₜ ; √σ²)

# Hamiltonian Monte Carlo 


    
# Problems that I have faced during GSOC

# Future improvements
