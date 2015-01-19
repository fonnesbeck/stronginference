Title: Burn-in, and Other MCMC Folklore
Date: 2014-08-09
Tags: bayesian, mcmc, pymc, python
Category: Statistics

<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

I have been slowly working my way through [The Handbook of Markov Chain Monte Carlo](http://amzn.to/mR9PVr), a compiled volume edited by Steve Brooks *et al.* that I picked up at last week's Joint Statistical Meetings. The first chapter is a primer on MCMC by [Charles Geyer](http://www.stat.umn.edu/~charlie/), in which he summarizes the key concepts of the theory and application of MCMC. In a particularly provocative passage, Geyer rips several of the traditional practices in setting up, running and diagnosing MCMC runs, including multi-chain runs, burn-in and sample-based diagnostics. Though they are applied regularly, these steps are simply heuristics that are applied to either aid in reaching or identifying the equilibrium distribution of the Markov chain. There are no guarantees on the reliability of any of them.

In particular, he questions the utility of burn-in:

> Burn-in is only one method, and not a particuarly good method, for finding a good starting point.
    
I can't disagree with this, though I have always viewed MCMC sampling (for most models that I have dealt with) as being cheap enough that there is little cost to simply throwing away thousands of them. I have often thrown away as many as the first 90 percent of my samples! However, as Geyer notes, there are better ways of getting your chain into a decent region of its support without throwing anything away.

One method is to use an approximation method on your model before applying MCMC. For example, the [maximum a posteriori (MAP)](http://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) estimate can be obtained using numerical optimization, then used as the initial values for an MCMC run. It turns out to be pretty easy to do in PyMC. For example, using the built-in bioassay example:

    :::python
    In [3]: from pymc.examples import gelman_bioassay

    In [4]: from pymc import MAP, MCMC

    In [5]: M = MAP(gelman_bioassay)

    In [6]: M.fit()

This yields MAP estimates for all the parameters in the model, which are less likely to be true modes as the complexity of the model increases, but are a pretty good bet to be a decent starting point for MCMC.

    :::python
    In [7]: M.alpha.value
    Out[7]: array(0.8465802225061101)
    
All that remains is to move these estimates into an MCMC sampler. While one could manually plug the values of each node into the model specification, its easiest just to extract the variables from the MAP estimator, and use them to instantiate an `MCMC` object:

    :::python
    In [8]: M.variables
    Out[8]: 
    set([<pymc.PyMCObjects.Stochastic 'alpha' at 0x10f78e810>,
         <pymc.PyMCObjects.Stochastic 'beta' at 0x10f78e910>,
         <pymc.PyMCObjects.Deterministic 'theta' at 0x10f78e9d0>,
         <pymc.distributions.Binomial 'deaths' at 0x10f78ea50>,
         <pymc.CommonDeterministics.Lambda 'LD50' at 0x10f78ec10>])

    In [9]: MC = MCMC(M.variables)

    In [10]: MC.sample(1000)
    Sampling: 100% [0000000000000000000000000000000000000000000000] Iterations: 1000
    
Notice that I did not pass a `burn` argument to MCMC, which defaults to zero. As is evident from the graphical output of the posteriors, this results in what appears to be a homogeneous chain, and which is hopefully already at its equilibrium distribution.

<img src="http://f.cl.ly/items/4513263v3x3n1T0m3o27/alpha.png" width="500">

<img src="http://f.cl.ly/items/1i0W0k1Q2S3h172E2v0b/beta.png" width="500">


What the MCMC practitioner fears is using a chain for inference that has not yet converged to its target distribution. Unfortunately, diagnostics cannot reliably alert you to this, nor does starting a model in several chains from disparate starting values guarantee this. There is also no magical threshold to distinguish convergence from pre-convergence regions in a MCMC trace. Geyer insists that only running chains for a very, very long time will inspire confidence:

> Your humble author has a dictum that the lease one can do is make an overnight run. ... If you do not make runs like that, you are simply not serious about MCMC.