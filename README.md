# Probabilistic Linear Solvers for Machine Learning

[![arXiv](https://img.shields.io/static/v1?logo=arxiv&logoColor=white&label=Preprint&message=1234.56789&color=B31B1B)](https://arxiv.org/abs/1234.56789)
[![NeurIPS](https://img.shields.io/static/v1?logo=material-design-icons&logoColor=white&label=NeurIPS&message=Proceedings&color=67458A)]()
[![ProbNum](https://img.shields.io/static/v1?logo=github&logoColor=white&label=ProbNum&message=Code&color=107d79)](https://github.com/probabilistic-numerics/probnum)
[![ProbNum](https://img.shields.io/static/v1?message=Docs&logo=read%20the%20docs&logoColor=white&label=ProbNum&color=blue)](https://probnum.readthedocs.io)

This repository contains additional resources such as experiments and data for the paper [Probabilistic Linear Solvers for Machine Learning]() by [Jonathan Wenger](https://jonathanwenger.netlify.app/) and [Philipp Hennig](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/personen/philipp-hennig/) published at NeurIPS 2020.

---

**Abstract:** Linear systems are the bedrock of virtually all numerical computation. Machine learning poses specific challenges for the solution of such systems due to their scale, characteristic structure, stochasticity and the central role of uncertainty in the field. Unifying earlier work we propose a class of probabilistic linear solvers which jointly infer the matrix, its inverse, and the solution from matrix-vector product observations. This class emerges from a fundamental set of desiderata which constrains the space of possible algorithms and recovers the method of conjugate gradients under certain conditions. We demonstrate how to incorporate prior spectral information in order to calibrate uncertainty and experimentally showcase the potential of such solvers for machine learning.

---

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

#### Kernel Gram Matrix Inversion
Calibration experiments on kernel matrices generated from the [flight delay dataset](/data/kernel_matrix_inversion/flight_delay_jan2020.zip) can be run in the following way. 

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

If you use this work in your research, please cite our paper:

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

## Acknowledgements

The authors gratefully acknowledge financial support by the European Research Council through ERC StG Action 757275 / PANAMA; the DFG Cluster of Excellence "Machine Learning - New Perspectives for Science", EXC 2064/1, project number 390727645; the German Federal Ministry of Education and Research (BMBF) through the Tübingen AI Center (FKZ: 01IS18039A); and funds from the Ministry of Science, Research and Arts of the State of Baden-Württemberg.

JW is grateful to the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for support.