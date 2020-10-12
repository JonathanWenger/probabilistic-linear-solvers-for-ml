# Probabilistic Linear Solvers for Machine Learning

This repository contains experiments and additional resources for the paper ["Probabilistic Linear Solvers for Machine Learning"]() (Jonathan Wenger and Philipp Hennig).

## Abstract

Linear systems are the bedrock of virtually all numerical computation. Machine learning poses specific challenges for the solution of such systems due to their scale, characteristic structure, stochasticity and the central role of uncertainty in the field. Unifying earlier work we propose a class of probabilistic linear solvers which jointly infer the matrix, its inverse, and the solution from matrix-vector product observations. This class emerges from a fundamental set of desiderata which constrains the space of possible algorithms and recovers the method of conjugate gradients under certain conditions. We demonstrate how to incorporate prior spectral information in order to calibrate uncertainty and experimentally showcase the potential of such solvers for machine learning.


<img align="left" src="https://raw.githubusercontent.com/JonathanWenger/probabilistic-linear-solvers-for-ml/main/figures/PLS_illustration.png" alt="PLS illustration" width="512" style="padding-right: 10px; padding left: 10px;" title="Illustration of a Probabilistic Linear Solver"/>



## Implementation

An implementation of our method is available as part of ProbNum

https://github.com/JonathanWenger/probabilistic-numerics/probnum

a Python framework for probabilistic numerics. You can find the API documentation of the probabilistic linear solver [here](https://probnum.readthedocs.io/en/latest/automod/probnum.linalg.problinsolve.html#probnum.linalg.problinsolve).

## Experiments

You can reproduce all experiments and plots shown in the paper in the following way. Notebooks producing plots and illustrations can be found in `/notebooks`. Calibration experiments performed using the [flight dataset from January 2020]() can be run by executing

```python

```

## Publication
If you use this work in your research, please cite the associated paper:

_"Probabilistic Linear Solvers for Machine Learning"_ ([PDF]()), Jonathan Wenger and Philipp Hennig

	@incollection{wenger2020problinsolve,
	  author        = {Jonathan Wenger and Philipp Hennig},
	  title         = {Probabilistic Linear Solvers for Machine Learning},
	  booktitle 	= {Advances in Neural Information Processing Systems 33}
	  year          = {2020},
	  keywords      = {probabilistic numerics, linear algebra, machine learning},
	  url           = {https://github.com/JonathanWenger/probabilistic-numerics/probnum}
	}

