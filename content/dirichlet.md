Title: Implementing Dirichlet processes for Bayesian semi-parametric models
Date: 2014-03-07
Tags: bayesian, pymc, mcmc, python
Category: Statistics

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

Semi-parametric methods have been preferred for a long time in survival analysis, for example, where the baseline hazard function is expressed non-parametrically to avoid assumptions regarding its form. Meanwhile, the use of non-parametric methods in Bayesian statistics is increasing. However, there are few resources to guide scientists in implementing such models using available software. Here, I will run through a quick implementation of a particular class of non-parametric Bayesian models, using PyMC.

Use of the term "non-parametric" in the context of Bayesian analysis is something of a misnomer. This is because the first and fundamental step in Bayesian modeling is to specify a *full probability model* for the problem at hand. It is rather difficult to explicitly state a full probability model without the use of probability functions, which are parametric. It turns out that Bayesian non-parametric models are not really non-parametric, but rather, are infinitely parametric.

A useful non-parametric approach for modeling random effects is the [Dirichlet process](http://en.wikipedia.org/wiki/Dirichlet_process). A Dirichlet process (DP), just like Poisson processes, Gaussian processes, and other processes, is a stochastic process. This just means that it comprises an indexed set of random variables. The DP can be conveniently thought of as a probability distribution of probability distributions, where the set of distributions it describes is infinite. Thus, an observation under a DP is described by a probability distribution that itself is a random draw from some other distribution. The DP (lets call it $G$) is described by two quantities, a baseline distribution $G_0$ that defines the center of the DP, and a concentration parameter $\\alpha$. If you wish, $G_0$ can be regarded as an *a priori* "best guess" at the functional form of the random variable, and $\\alpha$ as a measure of our confidence in our guess. So, as $\\alpha$ grows large, the DP resembles the functional form given by $G_0$.

To see how we sample from a Dirichlet process, it is helpful to consider the constructive definition of the DP. There are several representations of this, which include the Blackwell-MacQueen urn scheme, the stick-breaking process and the [Chinese restaurant process](http://en.wikipedia.org/wiki/Chinese_restaurant_process). For our purposes, I will consider the stick-breaking representation of the DP. This involves breaking the support of a particular variable into $k$ disjoint segments. The first break occurs at some point $x_0$, determined stochastically; the first piece of the notional "stick" is taken as the first group in the process, while the second piece is, in turn, broken at some selected point $x_1$ along its length. Here too, one piece is assigned to be the second group, while the other is subjected to the next break, and so on, until $k$ groups are created. Associated with each piece is a probability that is proportional to its length; these $k$ probabilities will have a Dirichlet distribution -- hence, the name of the process. Notice that $k$ can be infinite, making $G$ an infinite mixture.

We require two random samples to generate a DP. First, take a draw of values from the baseline distribution:

$$ \\theta_1, \\theta_2, \\ldots \\sim G_0 $$

then, a set of draws $v_1, v_2, \\ldots$ from a $\\text{Beta}(1,\\alpha)$ distribution. These beta random variates are used to assign probabilities to the $\\theta_i$ values, according to the stick-breaking analogy. So, the probability of $\\theta_1$ corresponds to the first "break", and is just $p_1 = v_1$. The next value corresponds to the second break, which is a proportion of the remainder from the first break, or $p_2 = (1-v_1)v_2$. So, in general:

$$ p_i = v_i \\prod_{j=1}^{i-1} (1 - v_j) $$

These probabilities correspond to the set of draws from the baseline distribution, where each of the latter are point masses of probability. So, the DP density function is:

$$ g(x) = \\sum_{i=1}^{\\infty} p_i I(x=\\theta_i) $$

where $I$ is the indicator function. So, you can see that the Dirichlet process is discrete, despite the fact that its values may be non-integer. This can be generalized to a mixture of continuous distributions, which is called a DP mixture, but I will focus here on the DP alone.

**Example: Estimating household radon levels**

As an example of implementing Dirichlet processes for random effects, I'm going to use the radon measurement and remediation example from [Gelman and Hill (2006)](http://amzn.to/gFfJbs). This problem uses measurements of [radon](http://en.wikipedia.org/wiki/Radon) (a carcinogenic, radioactive gas) from households in 85 counties in Minnesota to estimate the distribution of the substance across the state. This dataset has a natural hierarchical structure, with individual measurements nested within households, and households in turn nested within counties. Here, we are certainly interested in modeling the variation in counties, but do not have covariates measured at that level. Since we are more interested in the variation among counties, rather than the particular levels for each, a random effects model is appropriate. Whit Armstrong was kind enough to [code several parametrizations of this model in PyMC](https://github.com/armstrtw/pymc_radon), so I will use his code as a basis for implementing a non-parametric random effect for radon levels among counties.

In the original example from Gelman and Hill, measurements are modeled as being normally distributed, with a mean that is a hierarchical function of both a county-level random effect and a fixed effect that accounted for whether houses had a basement (this is thought to increase radon levels).

$$ y_i \\sim N(\\alpha_{j[i]} + \\beta x_i, \\sigma_y^2) $$

So, in essence, each county has its own intercept, but shares a slope among all counties. This can easily be generalized to both random slopes and intercepts, but I'm going to keep things simple, in order to focus in implementing a single random effect.

The constraint that is applied to the intercepts in Gelman and Hill's original model is that they have a common distribution (Gaussian) that describes how they vary from the state-wide mean.

$$ \\alpha_j \\sim N(\\mu_{\\alpha}, \\sigma_{\\alpha}^2) $$

This comprises a so-called "partial pooling" model, whereby counties are neither constrained to have identical means (full pooling) nor are assumed to have completely independent means (no pooling); in most applications, the truth is somewhere between these two extremes. Though this is a very flexible approach to accounting for county-level variance, one might be worried about imposing such a restrictive (thin-tailed) distribution like the normal on this variance. If there are counties that have extremely low or high levels (for whatever reason), this model will fit poorly. To allay such worries, we can hedge our bets by selecting a more forgiving functional form, such as [Student's t](http://en.wikipedia.org/wiki/Student's_t-distribution) or [Cauchy](http://en.wikipedia.org/wiki/Cauchy_distribution), but these still impose parametric restrictions (*e.g.* symmetry about the mean) that we may be uncomfortable making. So, in the interest of even greater flexibility, we will replace the normal county random effect with a non-parametric alternative, using a Dirichlet process.

One of the difficulties in implementing DP computationally is how to handle an infinite mixture. The easiest way to tackle this is by using a truncated Dirichlet process to approximate the full process. This can be done by choosing a size $k$ that is sufficiently large that it will exceed the number of point masses required. By doing this, we are assuming

$$ \\sum_{i=1}^{\\infty} p_i I(x=\\theta_i) \\approx \\sum_{i=1}^{N} p_i I(x=\\theta_i) $$

[Ohlssen et al. 2007](http://onlinelibrary.wiley.com/doi/10.1002/sim.2666/abstract) provide a rule of thumb for choosing $N$ such that the sum of the first $N-1$ point masses is greater than 0.99:

$$ N \\approx 5\\alpha + 2 $$

To be conservative, we will choose an even larger value (100), which we will call `N_dp`. The truncation makes implementation of DP in PyMC (or JAGS/BUGS) relatively simple.

We first must specify the baseline distribution and the concentration parameter. As we have no prior information to inform a choice for $\\alpha$, we will specify a uniform prior for it, with reasonable bounds:

	alpha = pymc.Uniform('alpha', lower=0.5, upper=10)

Though the upper bound may seem small for a prior that purports to be uninformative, recall that for large values of $\\alpha$, the DP will converge to the baseline distribution, suggesting that a continuous distribution would be more appropriate.

Since we are extending a normal random effects model, I will choose a normal baseline distribution, with vague hyperpriors:


	mu_0 = pymc.Normal('mu_0', mu=0, tau=0.01, value=0)
	sig_0 = pymc.Uniform('sig_0', lower=0, upper=100, value=1)
	tau_0 = sig_0 ** -2

	theta = pymc.Normal('theta', mu=mu_0, tau=tau_0, size=N_dp)

Notice that I have specified a uniform prior on the standard deviation, rather than the more common [gamma](http://en.wikipedia.org/wiki/Gamma_distribution)-distributed precision; for hierarchical models this is [good practice](http://ba.stat.cmu.edu/journal/2006/vol01/issue03/gelman.pdf). So, now we that we have `N_dp` point masses, all that remains is to generate corresponding probabilities. Following the recipe above:


	v = pymc.Beta('v', alpha=1, beta=alpha, size=N_dp)
	@pymc.deterministic
	def p(v=v):
	    """ Calculate Dirichlet probabilities """

	    # Probabilities from betas
	    value = [u*np.prod(1-v[:i]) for i,u in enumerate(v)]
	    # Enforce sum to unity constraint
	    value[-1] = 1-sum(value[:-1])

	    return value

This is where you really appreciate Python's [list comprehension](http://docs.python.org/tutorial/datastructures.html#list-comprehensions) idiom. In fact, were it not for the fact that we wanted to ensure that the array of probabilities sums to one, `p` could have been specified in a single line.

The final step involves using the Dirichlet probabilities to generate indices to the appropriate point masses. This is realized using a categorical mass function:


	z = pymc.Categorical('z', p, size=len(set(counties)))

These indices, in turn, are used to index the random effects, which are used as random intercepts for the model:


	a = pymc.Lambda('a', lambda z=z, theta=theta: theta[z])

Substitution of the above code into Gelman and Hill's original model produces reasonable results. The expected value of $\\alpha$ is approximately 5, as shown by the posterior output below:

![](http://dl.dropbox.com/u/233041/images/alpha.png)

Here is a random sample taken from the DP:

![](http://dl.dropbox.com/u/233041/images/dphist.png)

But is the model better? One metric for model comparison is the [deviance information criterion](http://en.wikipedia.org/wiki/Deviance_information_criterion) (DIC), which appears to strongly favor the DP random effect (smaller values are better):

	In [11]: M.dic
	Out[11]: 2138.7806225675804

	In [12]: M_dp.dic
	Out[12]: 1993.0894265799602

If you are interested in viewing the model code in its entirety, I have uploaded it to [my fork of Whit's code](https://github.com/fonnesbeck/pymc_radon/blob/master/radon_dp.py).