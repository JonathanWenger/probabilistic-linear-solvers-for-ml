# Probabilistic Linear Solvers for Machine Learning

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/1234.56789)
[![ProbNum](https://img.shields.io/readthedocs/probnum.svg?logo=github&logoColor=white&label=ProbNum)](https://github.com/probabilistic-numerics/probnum)
[![ProbNum Docs](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=ProbNum%20Docs)](https://probnum.readthedocs.io)

This repository contains additional resources such as experiments and data for the paper [Probabilistic Linear Solvers for Machine Learning]() by [Jonathan Wenger](https://jonathanwenger.netlify.app/) and [Philipp Hennig](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/personen/philipp-hennig/) published at NeurIPS 2020.

## Abstract

Linear systems are the bedrock of virtually all numerical computation. Machine learning poses specific challenges for the solution of such systems due to their scale, characteristic structure, stochasticity and the central role of uncertainty in the field. Unifying earlier work we propose a class of probabilistic linear solvers which jointly infer the matrix, its inverse, and the solution from matrix-vector product observations. This class emerges from a fundamental set of desiderata which constrains the space of possible algorithms and recovers the method of conjugate gradients under certain conditions. We demonstrate how to incorporate prior spectral information in order to calibrate uncertainty and experimentally showcase the potential of such solvers for machine learning.

<p align="center">
  <img src="https://raw.githubusercontent.com/JonathanWenger/probabilistic-linear-solvers-for-ml/main/figures/PLS_illustration.png" alt="PLS Illustration" width="800"/>
</p>


## Implementation

An open-source implementation of our method is available as part of <a href="https://github.com/probabilistic-numerics/probnum"><b>ProbNum</b></a>.

---

<a href="https://github.com/probabilistic-numerics/probnum"><img align="left" src="https://raw.githubusercontent.com/probabilistic-numerics/probnum/master/docs/source/img/pn_logo.png" alt="probabilistic numerics" width="120" style="padding-right: 5px; padding left: 5px;" title="Probabilistic Numerics in Python"/></a>**ProbNum implements probabilistic numerical methods in Python.** Such methods solve numerical problems from linear
algebra, optimization, quadrature and differential equations using _probabilistic inference_. This approach captures 
uncertainty arising from _finite computational resources_ and _stochastic input_. 

---

You can install ProbNum from the Python package index via:

```bash
pip install probnum
``` 

To get started check out the [tutorials](https://probnum.readthedocs.io/en/latest/tutorials/linear_algebra.html) on how to use the probabilistic linear solver within ProbNum and the [API documentation](https://probnum.readthedocs.io/en/latest/automod/probnum.linalg.problinsolve.html#probnum.linalg.problinsolve).

## Experiments

You can reproduce all experiments and plots shown in the paper with ProbNum v0.1.2.

```bash
pip install probnum==0.1.2
git clone git@github.com:JonathanWenger/probabilistic-linear-solvers-for-ml.git
cd probabilistic-linear-solvers-for-ml
```

#### Plots and Illustrations

Jupyter notebooks reproducing plots and illustrations are located in `./experiments/notebooks`. Simply install Jupyter and run the notebooks.

```bash
pip install jupyter
jupyter notebook
```

#### Calibration Experiments
Calibration experiments performed using the [flight delay dataset](/data/kernel_matrix_inversion/flight_delay_jan2020.zip) from January 2020 can be run in the following way. 

```bash
python experiments/scripts/kernel_matrix_inversion.py
```

#### Galerkin's Method

To apply the probabilistic linear solver to a discretization of a partial differential equation begin by running the associated notebook. To regenerate the mesh and resulting linear system using [FeNiCS](https://fenicsproject.org/) run:

```bash
pip install fenics
python experiments/scripts/poisson_pde.py
```


## Citation

If you use this work in your research, please cite the associated paper:

> Jonathan Wenger and Philipp Hennig. Probabilistic Linear Solvers for Machine Learning. *In Advances in Neural Information Processing Systems (NeurIPS)*, 2020

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
