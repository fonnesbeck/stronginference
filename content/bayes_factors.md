Title: Calculating Bayes factors with PyMC
Date: 2014-11-30
Tags: bayesian, pymc, mcmc, python
Category: Statistics
Slug: bayes-factors-pymc

<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

Statisticians are sometimes interested in comparing two (or more) models, with respect to their relative support by a particular dataset. This may be in order to select the best model to use for inference, or to weight models so that they can be averaged for use in multimodel inference. 

The [Bayes factor](http://en.wikipedia.org/wiki/Bayes_factor) is a good choice when comparing two arbitrary models, and the parameters of those models have been estimated. Bayes factors are simply ratios of _marginal_ likelihoods for competing models:

$$ \\text{BF}_{i,j} = \\frac{L(Y \\mid M_i)}{L(Y \\mid M_j)} = \\frac{\\int L(Y \\mid M_i,\\theta_i)p(\\mid \\theta_i \\mid M_i)d\\theta}{\\int L(Y \\mid M_j,\\theta_j)p(\\theta_j \\mid M_j)d\\theta} $$

While passingly similar to likelihood ratios, Bayes factors are calculated using likelihoods that have been integrated with respect to the unknown parameters. In contrast, likelihood ratios are calculated based on the maximum likelihood values of the parameters. This is an important difference, which makes Bayes factors a more effective means of comparing models, since it takes into account parametric uncertainty; likelihood ratios ignore this uncertainty. In addition, unlike likelihood ratios, the two models need not be nested. In other words, one model does not have to be a special case of the other.



Bayes factors are called Bayes factors because they are used in a Bayesian context by updating prior odds with information from data.

> Posterior odds = Bayes factor x Prior odds

Hence, they represent the evidence in the data for changing the prior odds of one model over another. It is this interpretation as a measure of evidence that makes the Bayes factor a compelling choice for model selection.

One of the obstacles to the wider use of Bayes factors is the difficulty associated with calculating them. While likelihood ratios can be obtained simply by the use of MLEs for all model parameters, Bayes factors require the integration over all unknown model parameters. Hence, for most interesting models Markov chain Monte Carlo (MCMC) is the easiest way to obtain Bayes factors.

Here's a quick tutorial on how to obtain Bayes factors from [PyMC](https://github.com/pymc-devs/pymc). I'm going to use a simple example taken from Chapter 7 of [Link and Barker (2010)](http://amzn.to/gGV2rK). Consider a short vector of data, consisting of 5 integers:

	:::python
    Y = array([0,1,2,3,8])

We wish to determine which of two functional forms best models this dataset. The first is a [geometric model](http://en.wikipedia.org/wiki/Geometric_distribution):

$$ f(x|p) = (1-p)^x p $$

and the second a [Poisson model](http://en.wikipedia.org/wiki/Poisson_distribution):

$$ f(x|\\mu) = \\frac{\\mu^x e^{-\\mu}}{x!} $$

Both describe the distribution of non-negative integer data, but differ in that the variance of Poisson data is equal to the mean, while the geometric model describes variance greater the mean. For this dataset, the sample variance would suggest that the geometric model is favored, but the sample is too small to say so definitively.

In order to calculate Bayes factors, we require both the prior and posterior odds:

>  Bayes factor = Posterior odds / Prior odds

The Bayes factor does not depend on the value of the prior model weights, but the estimate will be most precise when the posterior odds are the same. For our purposes, we will give 0.1 probability to the geometric model, and 0.9 to the Poisson model:

    :::python
    pi = (0.1, 0.9)

Next, we need to specify a latent variable, which identifies the true model (we don't believe either model is "true", but we hope one is better than the other). This is easily done using a categorical random variable, that identifies one model or the other, according to their relative weight.

    :::python
    true_model = Categorical("true_model", p=pi, value=0)

Here, we use the specified prior weights as the categorical probabilities, and the variable has been arbitrarily initialized to zero (the geometric model).

Next, we need prior distributions for the parameters of the two models. For the Poisson model, the expected value is given a uniform prior on the interval [0,1000]:
    
    :::python
    mu = Uniform('mu', lower=0, upper=1000, value=4)

This stochastic node can be used for the geometric model as well, though it needs to be transformed for use with that distribution:
    
    :::python
    p = Lambda('p', lambda mu=mu: 1/(1+mu))
    

Finally, the data are incorporated by specifying the appropriate likelihood. We require a mixture of geometric and Poisson likelihoods, depending on which value *true_model* takes. While BUGS requires an obscure trick to implement such a mixture, PyMC allows for the specification of arbitrary stochastic nodes: 

    :::python
    @observed
    def Ylike(value=Y, mu=mu, p=p, M=true_model):
        """Either Poisson or geometric, depending on M"""
        return geometric_like(value+1, p)*(M==0) or poisson_like(value, mu)

Notice that the function returns the geometric likelihood when M=0, or the Poisson model otherwise. Now, all that remains is to run the model, and extract the posterior quantities to calculate the Bayes factor.

Though we may be interested in the posterior estimate of the mean, all that we care about from a model selection standpoint is the estimate of *true_model*. At every iteration, the value of this parameter takes the value of zero for the geometric model and one for the Poisson. Hence, the mean (or median) will be an estimate of the probability of the Poisson model: 

    :::python
    In [11]: M.true_model.stats()['mean']
    
    Out[11]: 0.39654545454545453

So, the posterior probability that the Poisson model is true is about 0.4, leaving 0.6 for the geometric model. The Bayes factor in favor of the geometric model is simply:

    :::python
    In [18]: p_pois = M.true_model.stats()['mean']
    
    In [19]: ((1-p_pois)/p_pois) / (0.1/0.9)
    
    Out[19]: 13.696011004126548

This value can be interpreted as strong evidence in favor of the geometric model.

If you want to run the model for yourself, [you can download the code here](https://github.com/pymc-devs/pymc/wiki/BayesFactor).