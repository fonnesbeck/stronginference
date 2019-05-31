Title: PyMC 3.7: Making Data a First-Class Citizen
Date: 2019-05-31
Tags: bayesian, pymc, mcmc, python
Category: Statistics, Software
Slug: pymc37-release

This week featured the release of PyMC 3.7, which includes a slew of bug fixes and enhancements to help make building and fitting Bayesian models easier and more robust than ever. No fewer than 43 developers committed changes that became part of this release, so a big thanks goes out to all of them for their contributions. 

The number of new features in 3.7 is modest, but one in particular merits highlighting. Juan Martin Loyola generalized the `Minibatch` class, a wrapper for datasets that allows for stochastic gradient calculations in variational inference, to more formally integrate datasets into model specification. Specifically, the `Data` class is a container for data that endows it with many of the attributes of other PyMC3 variable objects. What this does is formally incorporate your data into the model graph, as a deterministic node. 

Behind the scenes, `Data` converts data arrays into Theano `shared` variables which, as the name implies, is a variable whose value is shared between functions in which it is used. This allows the values to be changed between model runs, without having to re-specify the model itself. Note that this is exactly how `Minibatch` works, though for `Minibatch` the values are changed within a run of variational inference fitting. There are two general scenarios where using the `Data` wrapper will be helpful: making predictions on a new subset of data after model fitting, and running the same model on multiple datasets. 

Let's look at a quick example to see how it works. 

Here's a simple logistic regression model for analyzing baseball data that tries to estimate the probability that a batter swings and misses at a curveball as a function of the rate at which the ball spins (it is thought that more effective pitches are related to higher spin rates). 

```python
curveball_data.head()
```

![Curveball spin data](https://d.pr/i/4ubok7+)

We can wrap the spin rate column in `Data` and give it a name (let's call it "spin") and do the same for the binary outcome ("miss"). We then reference these objects in the model where we otherwise would have passed NumPy `array` or pandas `Series` objects.

```python
with pm.Model() as curve_spin_model:

    spin = pm.Data('spin', curveball_data['spin_rate'])
    β = pm.Normal('β', shape=2)
    
    θ = β[0] + β[1]*(spin/1000)
    
    swing_miss = pm.Data('swing_miss', curveball_data['miss'])
    miss = pm.Bernoulli('miss', pm.invlogit(θ), observed=swing_miss)

    trace = pm.sample(1000, tune=6000, cores=2)
```

Notice the positive estimate for $\beta[0]$, which suggests higher spin rates lead to more swinging strikes.

```python
az.summary(trace, var_names=['β'])
```

![Parameter estimates](https://d.pr/i/Px9f9j+)

Now, let's say we have pitchers with known spin rates on their curve balls. We can use `sample_posterior_predictive` to predict the associated miss rate, but first we use `set_data` within the model context to swap in the new data. Nothing else needs to change!

```python
more_curves = [1810, 2225, 3015]
with curve_spin_model:
    pm.set_data({'spin': more_curves})
    post_pred = pm.sample_posterior_predictive(trace, samples=1000)
```

So, it looks like increasing spin rate from 1810 rpm to 3015 rpm could result in about an 11 percent increase in miss rate!

```python
post_pred['miss'].mean(0)
```

```
array([0.379, 0.428, 0.481])
```

Additionally, `set_data` can be used to pass batches of independent data to models for sequential fitting. Let's extend the above example to the case where there are multiple years of data:

![Curveball annual data](https://d.pr/i/nfoswZ+)

We might then want separate, annual estimates of curveball effectiveness as a function of spin. This is a matter of looping over the years, and passing the subsets to the model prior to each fit.

```python
traces = {}

for year in years:
    with curve_spin_model:
        year_data = curveball_data.query('year==@year')
        pm.set_data({'spin': year_data.spin_rate, 
                    'swing_miss': year_data.miss})
        traces[year] = pm.sample(1000, tune=2000,
                                 cores=2, progressbar=False)
```

Pretty slick.

There are numerous other smaller changes in PyMC 3.7. Of note, following much discussion the `sd` argument has been renamed to `sigma` for scale parameters. There is conflicting opinion regarding the use of Greek variable names for distribution parameters, but at least our convention is now consistent, in that we use `mu` and `sigma` rather than `mu` and `sd` (though `sd` will still be accepted in order to retain backward-compatibility). 

Another nice enhancement is a fix to the `Mixture` class that allows it to work properly with multidimensional or multivariate distributions. In previous releases, model fitting would often work, but any sort of predictive sampling would fail.

If you are a fan of the GLM submodule, you will be happy to learn that it is now a little more flexible. Any model strings passed to `from_formula` can now extract variables from calling scope. For example:

```python
df = pd.DataFrame(dict(y=np.array([2, 4, 2, 0, 4])))

with pm.Model() as glm:
    x = np.array([0.3, 1.1, 0.6, -0.2, 3.3])
    pm.glm.GLM.from_formula('y ~ x', data=df, family=pm.glm.families.Poisson())
```

Prior to this release, all of the data had to be shoehorned into the `DataFrame` passed to `from_formula`.

A complete list of changes can be found on our [GitHub repository](https://github.com/pymc-devs/pymc3/releases/tag/v3.7). The 3.7 release is recommended for all users. You can install it now, either via `pip`:

    pip install -U pymc3

or `conda`, if you are using Anaconda's Python distribtuion:

    conda install -c conda-forge pymc3

Thanks once again to the entire PyMC3 development team, and to diligent users who reported (and sometimes fixed) bugs. Happy sampling!