# Bayesian_Examples.jl
Examples of my project for Google Summer of Code

# Literature Overview
  * Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
  * Drovandi, C. C., Pettitt, A. N., & Lee, A. (2015). Bayesian indirect inference using a parametric auxiliary model. 
  * Rayner, G. D. and MacGillivray, H. L. (2002). Numerical maximum likelihood estimation for the g-and-k and generalized g-and-h distributions. Stat. Comput. 12 57–75.

# GSOC 2017 project: Bayesian estimation using Random Walk Metropolis-Hastings and Dynamic Hamiltonian Monte Carlo methods

This summer I have had the opportunity to participate in the Google Summer of Code program. My project was in the Julia language and the main goal was to  with the purpose to use different methods to estimate models in order to improve efficiency. 

Under the mentorship of Tamás K. Papp, I completed a major revision of Bayesian estimation methods using Indirect Inference (II) and Hamiltonian Monte Carlo. I also got familiar with using git, opening issues, creating a repository among others. 

Hopefully, by the end of this post, I will manage to introduce these methods a little bit better and more extensively.

# Bayesian Indirect Inference Using a Parametric Auxiliary Model

Facing an intractable model, we have the option to use an alternative auxiliary model to extract and estimate the parameters of interest. These alternative models should be easier to deal with. Drovandi et al. introduced a collection of parametric Bayesian Indirect Inference (pBII) methods, I focused on the parametric Bayesian Inidrect Likelihood (pBIL) method. The pBIL uses the likelihood of the auxiliary model as a substitute for the intractable likelihood. The pBIL does not compare summary statistics, instead works in the following way: 

First the data is generated, once we have the data, we can estimate the parameters of the auxiliary model. Then, the estimated parameters are put into the auxiliary likelihood with the observed/generated data. Afterwards we can use this likelihood in our chosen Bayesian method i.e. MCMC. 

In the first stage of my project I coded two models from Drovandi et al. using pBIL. Aftar calculating the likelihood of the auxiliary model, I used a Random Walk Metropolis-Hastings MCMC to sample from the target distribution. It resulted in [HamiltonianABC](https://github.com/tpapp/HamiltonianABC.jl/) (collaboration with Tamás K. Papp).

**First model** \
  * The true model is y ∼ Exponential(λ), IID, where λ is the scale. \
  * The auxiliary model is y ∼ N(μ, σ²), with statistics ϕ = (μ, σ). \
  * The prior is λ ∼ Uniform(A,B) prior. 

This model and the estimation worked fine.

**Second model** \
  * The true model was the g-and-k quantile function described by Rayner and MacGillivray (2002). \
  * The auxiliary model was a three component normal mixture model. 

We faced serious problems with this model. First of all, I coded the MLE of the finite component normal mixture model, but 









