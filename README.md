# Probabilistic Linear Solvers for Machine Learning

This repository contains additional resources such as experiments for the paper ["Probabilistic Linear Solvers for Machine Learning"]() by Jonathan Wenger and Philipp Hennig published at NeurIPS 2020.

## Abstract

Linear systems are the bedrock of virtually all numerical computation. Machine learning poses specific challenges for the solution of such systems due to their scale, characteristic structure, stochasticity and the central role of uncertainty in the field. Unifying earlier work we propose a class of probabilistic linear solvers which jointly infer the matrix, its inverse, and the solution from matrix-vector product observations. This class emerges from a fundamental set of desiderata which constrains the space of possible algorithms and recovers the method of conjugate gradients under certain conditions. We demonstrate how to incorporate prior spectral information in order to calibrate uncertainty and experimentally showcase the potential of such solvers for machine learning.

<p align="center">
  <img src="https://raw.githubusercontent.com/JonathanWenger/probabilistic-linear-solvers-for-ml/main/figures/PLS_illustration.png" alt="PLS Illustration" width="800"/>
</p>


## Implementation

An open-source implementation of our method is available as part of <a href="https://github.com/probabilistic-numerics/probnum"><b>ProbNum</b></a>.

---

<a href="https://probnum.readthedocs.io"><img align="left" src="https://raw.githubusercontent.com/probabilistic-numerics/probnum/master/docs/source/img/pn_logo.png" alt="probabilistic numerics" width="120" style="padding-right: 5px; padding left: 5px;" title="Probabilistic Numerics in Python"/></a>**ProbNum implements probabilistic numerical methods in Python.** Such methods solve numerical problems from linear
algebra, optimization, quadrature and differential equations using _probabilistic inference_. This approach captures 
uncertainty arising from _finite computational resources_ and _stochastic input_. 

---

You can install ProbNum using from the Python package index via:

```bash
pip install probnum==0.1.2
``` 

To get started check out the [tutorials](https://probnum.readthedocs.io/en/latest/tutorials/linear_algebra.html) on how to use the probabilistic linear solver within ProbNum and the [API documentation](https://probnum.readthedocs.io/en/latest/automod/probnum.linalg.problinsolve.html#probnum.linalg.problinsolve).

## Experiments

You can reproduce all experiments and plots shown in the paper in the following way. Notebooks producing plots and illustrations can be found in `/notebooks`. Calibration experiments performed using the [flight dataset from January 2020](/data/kernel_matrix_inversion/flight_delay_jan2020.zip) can be run by executing

```python

```

## Citation

If you use this work in your research, please cite the associated paper:

_"Probabilistic Linear Solvers for Machine Learning"_ ([PDF]()), Jonathan Wenger and Philipp Hennig

```bibtex
@incollection{wenger2020problinsolve,
  author        = {Jonathan Wenger and Philipp Hennig},
  title         = {Probabilistic Linear Solvers for Machine Learning},
  booktitle 	= {Advances in Neural Information Processing Systems 33}
  year          = {2020},
  keywords      = {probabilistic numerics, linear algebra, machine learning},
  url           = {https://github.com/JonathanWenger/probabilistic-numerics/probnum}
}
```
