Title: The First Release of PyMC3
Date: 2017-01-10
Tags: bayesian, pymc, mcmc, python, variational inference
Category: Statistics
Slug: pymc3-release

![pymc3](http://d.pr/i/lJ7d+)

On Monday morning the PyMC dev team pushed the first release of PyMC3, the culmination of over 5 years of collaborative work. We are very pleased to be able to provide a stable version of the package to the Python scientific computing community. For those of you unfamiliar with the history and progression of this project, PyMC3 is a complete re-design and re-write of the PyMC code base, which was primarily the product of the vision and work of John Salvatier. While PyMC 2.3 is still actively maintained and used (I continue to work with it in a number of project myself), this new incarnation allows us to be able to provide newer methods for Bayesian computation to a degree that would have been impossible impossible previously. 

While PyMC2 relied on Fortran extensions (via `f2py`) for most of the computational heavy-lifting, PyMC3 leverages [Theano](http://deeplearning.net/software/theano/), a library from the LISA lab for array-based expression evaluation, to perform its computation. What this provides, above all else, is fast automatic differentiation, which is at the heart of the gradient-based sampling and optimization methods currently providing inference for probabilistic programming. While the addition of Theano adds a level of complexity to the development of PyMC, fundamentally altering how the underlying computation is performed, we have worked hard to maintain the elegant simplicity of the original PyMC model specification syntax. Since the beginning (over 13 years ago now!), we have tried to provide a simple, black-box interface to model-building, in the sense that the user need only concern herself with the modeling problem at hand, rather than with the underlying computer science. 

As a point of comparison, here is what a simple hierarchical model (taken from [Gelman *et al.*'s book](https://www.amazon.com/Bayesian-Analysis-Chapman-Statistical-Science/dp/1439840954)) looked like under PyMC 2.3:

```python
# Priors
alpha = Normal('alpha', 0, 0.01)
beta = Normal('beta', 0, 0.01)
 
# Transformed variables
theta = Lambda('theta', lambda a=alpha, b=beta, d=dose: invlogit(a + b * d))
 
# Data likelihood
deaths = Binomial('deaths', n=n, p=theta, value=array([0,1,3,5]), observed=True) 

# Instantiate a sampler, and run
M = MCMC(locals())
M.sample(10000, burn=5000)
```

and here is the same model in PyMC3:

```python
with Model() as bioassay_model:

	alpha = Normal('alpha', 0, sd=100)
	beta = Normal('beta', 0, sd=100)
			
	theta = invlogit(alpha + beta*dose)

	deaths = Binomial('deaths', n=n, p=theta, observed=array([0,1,3,5]))
	
	trace = sample(2000)
```

If anything, the model specification has simplified, for the majority of models.

Though the version 2 and version 3 models are superficially similar (by design), there are very different things happening underneath when `sample`is called in either case. By default, the PyMC3 model will use a form of gradient-based MCMC sampling, a self-tuning form of Hamiltonian Monte Carlo, called NUTS.  Gradient based methods serve to drastically improve the efficiency of MCMC, without the need for running long chains and dropping large portions of the chains due to lack of convergence. Rather than conditionally sampling each model parameter in turn, the NUTS algorithm walks in k-space (where k is the number of model parameters), simultaneously updating all the parameters as it leap-frogs through the parameter space. Models of moderate complexity and size that would normally require 50,000 to 100,000 iterations now typically require only 2000-3000.

When we run the PyMC3 version of the model above, we see this:

```
Auto-assigning NUTS sampler...
Initializing NUTS using advi...
Average ELBO = -6.2597: 100%|████████████████████████████████████████| 200000/200000 [00:11<00:00, 16873.12it/s]
Finished [100%]: Average ELBO = -6.27
100%|██████████████████████████████████████████████████████████████████████| 2000/2000 [00:02<00:00, 928.24it/s]
```

Unless specified otherwise, PyMC3 will assign the NUTS sampler to all the variables of the  model. This happens here because our model contains only *continuous* random variables; NUTS will not work with discrete variables because it is impossible to obtain gradient information from them. Discrete variables are assigned the `Metropolis`sampling algorithm (*step method*, in PyMC parlance). The next thing that happens is that the variables' initial values are assigned using Automatic Differentiation Variational Inference (ADVI). This is an approximate Bayesian inference algorithm that we have added to PyMC — more on that later. Though it can be used for inference in its own right, here we are using it merely to find good starting values for NUTS (in practice, this is important for getting NUTS to run well). Its an excessive step for small models like this, but it is the default behavior, designed to try and guarantee a good MCMC run.

Another nice innovation includes some new plotting functions for visualizing the posterior distributions obtained with the various estimation methods. Let's look at the regression parameters from our fitted model:

```python
plot_posterior(trace, varnames=['beta', 'alpha'])
```

![posterior plot](http://d.pr/i/41uE+)

`plot_posterior` generates histograms of the posterior distribution that is annotated with summary statistics of interest, in the style of [John Kruschke's book](https://www.amazon.com/Doing-Bayesian-Data-Analysis-Tutorial/dp/0123814855). This is just one of several options for visualizing output.

The addition of variational inference (VI) methods in version 3.0 is a transformative change to the sorts of problems you can tackle with PyMC3. I showed it being used to intialize a model that was ultimately fit using MCMC, but variational inference can be used as a tool for obtaining statistical inference in its own right. Just as MCMC approximates a complex posterior by drawing dependent samples from its posterior distribution, variational inference performs an approximation by replacing the true posterior with a more tractable form, then iteratively changes the approximation so that it resembles the posterior distribution as closely as it can, in terms of the *information distance* between the two distributions. Where MCMC uses sampling, VI uses optimization to estimate the posterior distribution. The benefit to you in doing this is that Bayesian models informed by very large datasets can be fit in a reasonable amount of time (MCMC notoriously scales poorly with data size); the drawback is that you only get an approximation to the posterior, and that appoximation can be unacceptably poor for some applications. Nevertheless, improvements to variational inference methods continue to roll in, and [some have the potential to drastically improve the quality of the approximation](https://arxiv.org/abs/1505.05770). The key advance that allowed PyMC3 to implement variational methods was the development of automated algorithms for specifying a variational approximation generally, across a wide variety of models. In particular, Alp Kucukelbir and colleagues' introduction of [Automatic Differentiation Variational Inference (ADVI)](https://arxiv.org/abs/1603.00788) two years ago, combined with the ability of Theano to provide automatic differentiation of Python models, made VI relatively easy to apply to arbitrary models (again, assuming the model variables are continuous). Here it is, in action, fitting the same model we used NUTS to estimate before:

```python
with model:
    advi_fit = advi(n=10000)
```

```
Average ELBO = -6.2765: 100%|████████████████████████████████████████████████| 100000/100000 [00:05<00:00, 17072.45it/s]
Finished [100%]: Average ELBO = -6.2835
```

ADVI returns the means and standard deviations of the approximating distribution after it has converged to the best approximation. These values can be used to sample from the disribtution:

```python
with model:
    trace = sample_vp(advi_fit, 10000)
```



![advi samples](http://d.pr/i/IT5O+)

As we push past the PyMC3 3.0 release, we have a number of innovations either under development or in planning. For example, in order to improve the quality of approximations using variational inference, we are looking at implementing methods that transform the approximating density to allow it to represent more complicated distributions, such as the application of normalizing flows to ADVI; this work is being led by Taku Yoshioka. Thomas Wiecki is currently working on adding Stein Variational Gradient Descent to the suite of VI algorithms, which should allow much larger datasets to be fit to PyMC models. To more easily accommodate the number of different VI algorithms that are being developed, Maxim Kochurov is leading the development of a flexible base class for variational methods that will unify their interfaces.  Work is also underway to allow PyMC3 to take advantage of computation on GPUs, something that Theano allows us to do, but requires some engineering to allow it to work generally. These are just a few notable enhancements, along with all of the incremental but steady improvement throughout the code base.

When I began the PyMC project as a postdoctoral fellow [back in 2003](https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_2003), it was intended only as a set of functions and classes for personal use, to simplify the business of building and iterating through sets of models. At the time, the world of Bayesian computation was dominated by WinBUGS, a truly revolutionary piece of software that made hierarchical modeling and MCMC available to applied statisticians and other scientists who would otherwise been unable to consider these approaches. All the same, the BUGS language was not ideal for all problems and workflows, so if you needed something else you were forced to write your own software. We live in a very different scientific computing world today; for example, there are, as of this writing, no fewer than six libraries for building Gaussian process models in Python! The ecosystem for probabilistic programming and Bayesian analysis is rich today, and becoming richer every month, it seems.

I'd like to take the opportunity now to thank the ever-changing and -growing PyMC development team for all of their hard work over the years. I've been truly awe-stricken by the level of talent and degree of comittment that the project has attracted over the years. Some contributors added value to the project for very short intervals,  perhaps in order to facilitate the completion of their own work, and others have stuck around through multiple releases, not only implementing exciting new functionality, but also taking on more mundane chores like squashing bugs and refactoring old code. Of course, every bit helps. Thanks again.

Finally, I'd like to extend an invitation to all who are interested (or just curious) to come on board and contribute. Now is an exciting time to be a part of the team, with novel methodological innovations in Bayesian modeling arriving at such a rapid pace, and with data science coming into its own as a field. We welcome contributions to all aspects of the project: code development, [issue](https://github.com/pymc-devs/pymc3/issues) resolution, [documentation](http://pymc-devs.github.io/pymc3/) writing—simply trying out PyMC3 on your own problem and reporting what does and doesn't work is even a great way to get involved. It doesn't take much to get started! 