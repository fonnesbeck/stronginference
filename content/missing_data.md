Title: Automatic Missing Data Imputation with PyMC
Date: 2013-08-18
Tags: pymc, mcmc, python
Category: Statistics
Slug: missing-data-imputation

<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

A distinct advantage of using Bayesian inference is in its universal application of probability models for providing inference. As such, all components of a Bayesian model are specified using probability distributions for either describing a sampling model (in the case of observed data) or characterizing the uncertainty of an unknown quantity. This means that missing data are treated the same as parameters, and so imputation proceeds very much like estimation. When using Markov chain Monte Carlo (MCMC) to fit Bayesian models it usually requires only a few extra lines of code to impute missing values, based on the sampling distribution of the missing data, and associated (usually unknown) parameters. Using [PyMC built from the latest development code][pymc], missing data imputation can be done automatically.

## Types of Missing Data ##

The appropriate treatment of missing data depends strongly on how the data came to be missing from the dataset. These mechanisms can be broadly classified into three groups, according to how much information and effort is required to deal with them adequately.

### Missing completely at random (MCAR) ###

If data are MCAR, then the probability of that any given datum is missing is equal over the whole dataset. In other words, each datum that is present had the same probability of being missing as each datum that is absent. This implies that ignoring the missing data will not bias inference.

### Missing at random (MAR) ###

MAR allows for data to be missing according to a random process, but is more general than MCAR in that all units do not have equal probabilities of being missing. The constraint here is that missingness may only depend on information that is fully observed. For example, the reporting of income on surveys may vary according to some measured factor, such as age, race or sex. We can thus account for heterogeneity in the probability of reporting income by controlling for the measured covariate in whatever model is used for infrence.

### Missing not at random (MNAR) ###

When the probability of missing data varies according to information that is not available, this is classified as MNAR. This can either be because suitable covariates for explaining missingness have not been recorded (or are otherwise unavailable) or the probability of being missing depends on the value of the missing datum itself. Extending the previous example, if the probability of reporting income varied according to income itself, this is missing not at random.

In each of these situations, the missing data may be imputed using a sampling model, though in the case of missing not at random, it may be difficult to validate the assumptions required to specify such a model. For the purposes of quickly demonstrating automatic imputation in PyMC, I will illustrate using data that is MCAR.

## Implementing imputation in PyMC ##

One of the recurring examples in the PyMC documentation is the coal mining disasters dataset from [Jarrett 1979][jarrett79]. This is a simple longitudinal dataset consisting of counts of coal mining disasters in the U.K. between 1851 and 1962. The objective of the analysis is to identify a switch point in the rate of disasters, from a relatively high rate early in the time series to a lower one later on. Hence, we are interested in estimating two rates, in addition to the year after which the rate changed.

In order to illustrate imputation, I have randomly replaced the data for two years with a missing data placeholder value, -999:

    disasters_array =   np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                       3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                       2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                       1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                       0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                       3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                       0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
                       
Here, the `np` prefix indicates that the `array` function comes from the [Numpy][numpy] module. PyMC is able to recognize the presence of missing values when we use Numpy's MaskedArray class to contain our data. The masked array is instantiated via the `masked_array` function, using the original data array and a boolean mask as arguments: 

        masked_values = np.ma.masked_array(disasters_array,
 		mask=disasters_array==-999)
    
Of course, my use of -999 to indicate missing data was entirely arbitrary, so feel free to use any appropriate value, so long as it can be identified and masked (obviously, small positive integers would not have been appropriate here). Let's have a look at the masked array:

    masked_array(data = [4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6 3 3 5 4 5 3 1 4 
		4 1 5 5 3 4 2 5 2 2 3 4 2 1 3 -- 2 1 1 1 1 3 0 0 1 0 1 1 0 0 3 1 
		0 3 2 2 0 1 1 1 0 1 0 1 0 0 0 2 1 0 0 0 1 1 0 2 3 3 1 -- 2 1 1 1 
		1 2 4 2 0 0 1 4 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1], 
		mask = [False False False False False False False False False 
		False False False False False False False False False False False
	 	False False False False False False False False False False False
	    False False False False False False False False True False False 
		False False False False False False False False False False False
	    False False False False False False False False False False False
	    False False False False False False False False False False False
	    False False False False False False False False True False False 
	    False False False False False False False False False False False
	    False False False False False False False False False False False 
	    False False False],
        fill_value = 999999)

Notice that the placeholder values have disappeared from the data, and the array has a `mask` attribute that identifies the indices for the missing values.

Beyond the construction of a masked array, there is nothing else that needs to be done to accommodate missing values in a PyMC model.
       
First, we need to specify prior distributions for the unknown parameters, which I call `switch` (the switch point), `early` (the early mean) and `late` (the late mean). An appropriate non-informative prior for the switch point is a discrete uniform random variable over the range of years represented by the data. Since the rates must be positive, I use identical weakly-informative exponential distributions:

    # Switchpoint
    switch = DiscreteUniform('switch', lower=0, upper=110)
    # Early mean
    early = Exponential('early', beta=1)
    # Late mean
    late = Exponential('late', beta=1)
    
The only tricky part of the model is assigning the appropriate rate parameter to each observation. Though the two rates and the switch point are stochastic, in the sense that we have used probability models to describe our uncertainty in their true values, the membership of each observation to either the early or late rate is a deterministic function of the stochastics. Thus, we set up a deterministic node that assigns a rate to each observation depending on the  location of the switch point at the current iteration of the MCMC algorithm:

    @deterministic
    def rates(s=switch, e=early, l=late):
        """Allocate appropriate mean to time series"""
        out = np.empty(len(disasters_array))
        # Early mean prior to switchpoint
        out[:s] = e
        # Late mean following switchpoint
        out[s:] = l
        return out

Finally, the data likelihood comprises the annual counts of disasters being modeled as Poisson random variables, conditional on the parameters assigned in the `rates` node above. The masked array is specified as the value of the stochastic node, and flagged as data via the `observed` argument.

    disasters = Poisson('disasters', mu=rates, value=masked_values, observed=True)
    
If we run the model, then query the `disasters` node for posterior statistics, we can obtain a summary of the estimated number of disasters in both of the missing years.

    In [9]: DisasterModel.disasters.stats()
    Out[9]: 
    {'95% HPD interval': array([[ 0.,  6.],
           [ 0.,  3.]]),
     'mc error': array([ 0.11645149,  0.03479713]),
     'mean': array([ 2.2246,  0.91  ]),
     'n': 5000,
     'quantiles': {2.5: array([ 0.,  0.]),
                   25: array([ 1.,  0.]),
                   50: array([ 2.,  1.]),
                   75: array([ 3.,  1.]),
                   97.5: array([ 7.,  3.])},
     'standard deviation': array([ 1.88206133,  0.92536479])}

Clearly, this is a rather trivial example, but it serves to illustrate how easy it can be to deal with missing values in PyMC. Though not applicable here, it would be similarly easy to handle MAR data, by constructing a data likelihood whose parameter(s) is a function of one or more covariates. 

Automatic imputation is a new feature in PyMC, and is currently available only in the [development codebase][pymc-github]. It will hopefully appear in the feature set of a future release.

[pymc]: http://github.com/pymc-devs/pymc "PyMC on GitHhub"
[gelman04]: http://www.stat.columbia.edu/~gelman/book/ "Bayesian Data Analysis, by Gelman, Carlin, Stern, and Rubin (1995, 2004)"
[jarrett79]: http://biomet.oxfordjournals.org/cgi/content/short/66/1/191 "Jarrett RG (1979). A Note on the Intervals Between Coal Mining Disasters. Biometrika, 66, 191–193."
[numpy]: http://numpy.scipy.org/
[pymc-github]: http://github.com/pymc-devs/pymc