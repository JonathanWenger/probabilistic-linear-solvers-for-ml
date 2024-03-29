# Probabilistic Linear Solvers for Machine Learning

[![arXiv](https://img.shields.io/static/v1?logo=arxiv&logoColor=white&label=Preprint&message=2010.09691&color=B31B1B)](https://arxiv.org/abs/2010.09691)
[![NeurIPS](https://img.shields.io/static/v1?logo=material-design-icons&logoColor=white&label=NeurIPS&message=Proceedings&color=67458A)](https://papers.nips.cc/paper/2020/hash/4afd521d77158e02aed37e2274b90c9c-Abstract.html)
[![ProbNum](https://img.shields.io/static/v1?logo=github&logoColor=white&label=ProbNum&message=Code&color=107d79)](https://github.com/probabilistic-numerics/probnum)
[![ProbNum](https://img.shields.io/static/v1?message=Docs&logo=read%20the%20docs&logoColor=white&label=ProbNum&color=blue)](https://probnum.readthedocs.io)

This repository contains additional resources such as experiments and data for the paper [Probabilistic Linear Solvers for Machine Learning](https://arxiv.org/abs/2010.09691) by [Jonathan Wenger](https://jonathanwenger.netlify.app/) and [Philipp Hennig](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/personen/philipp-hennig/) published at NeurIPS 2020.

---

**Abstract:** Linear systems are the bedrock of virtually all numerical computation. Machine learning poses specific challenges for the solution of such systems due to their scale, characteristic structure, stochasticity and the central role of uncertainty in the field. Unifying earlier work we propose a class of probabilistic linear solvers which jointly infer the matrix, its inverse and the solution from matrix-vector product observations. This class emerges from a fundamental set of desiderata which constrains the space of possible algorithms and recovers the method of conjugate gradients under certain conditions. We demonstrate how to incorporate prior spectral information in order to calibrate uncertainty and experimentally showcase the potential of such solvers for machine learning.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/JonathanWenger/probabilistic-linear-solvers-for-ml/main/figures/PLS_illustration.png" alt="PLS Illustration" width="800"/>
</p>


## Implementation

An open-source implementation of our method is available as part of <a href="https://github.com/probabilistic-numerics/probnum"><b>ProbNum</b></a>.

---

<a href="https://github.com/probabilistic-numerics/probnum"><img align="left" src="https://github.com/probabilistic-numerics/probnum/blob/37b51b163d55c195d05b38d2eeed19db997fe082/docs/source/assets/img/logo/probnum_logo.png" alt="probabilistic numerics" width="90" style="padding-right: 5px; padding left: 5px;" title="Probabilistic Numerics in Python"/></a>**ProbNum implements probabilistic numerical methods in Python.** Such methods solve numerical problems from linear
algebra, optimization, quadrature and differential equations using _probabilistic inference_. This approach captures 
uncertainty arising from _finite computational resources_ and _stochastic input_. 

---

You can install ProbNum from the Python package index via:

```bash
pip install probnum==0.1.21
``` 

To get started check out the [tutorials](https://probnum.readthedocs.io/en/latest/tutorials/linear_algebra.html) on how to use the probabilistic linear solver within ProbNum and the [API documentation](https://probnum.readthedocs.io/en/latest/automod/probnum.linalg.problinsolve.html#probnum.linalg.problinsolve).

## Experiments

You can reproduce all experiments and plots shown in the paper using [ProbNum v0.1.21](https://probnum.readthedocs.io/en/v0.1.21/).

```bash
git clone git@github.com:JonathanWenger/probabilistic-linear-solvers-for-ml.git
cd probabilistic-linear-solvers-for-ml
pip install -r requirements.txt
```

#### Plots and Illustrations

Jupyter notebooks reproducing plots and illustrations are located in `./experiments/notebooks`. Simply install Jupyter and run the notebooks.

```bash
jupyter notebook
```

#### Kernel Gram Matrix Inversion
Calibration experiments on kernel matrices generated from the [flight delay dataset](/data/kernel_matrix_inversion/flight_delay_jan2020.zip) can be run in the following way. 

```bash
cd experiments/scripts
python kernel_matrix_inversion.py
```

All calibration strategies reduced overconfidence of the solver for the evaluated kernel matrices.

<p align="center">
  <img src="https://github.com/JonathanWenger/probabilistic-linear-solvers-for-ml/blob/main/figures/calibration_experiment_table.png" alt="calibration_experiment" width="400"/>
</p>

#### Galerkin's Method

To apply the probabilistic linear solver to a discretization of a partial differential equation run the associated Jupyter notebook in `./experiments/notebooks`.

If you wish to regenerate the mesh and resulting linear system via Galerkin's method yourself, begin by installing [FeNiCS](https://fenicsproject.org/download). On Ubuntu you can simply install from the package management system.
```bash
sudo apt-get install fenics
```

You can now generate the coarse and fine meshes and associated linear systems for the Dirichlet equation via:

```bash
cd experiments/scripts
python poisson_pde.py -r 6
python poisson_pde.py -r 128
```

## Citation

If you use this work in your research, please cite our paper:

> Jonathan Wenger and Philipp Hennig. Probabilistic Linear Solvers for Machine Learning. *In Advances in Neural Information Processing Systems (NeurIPS)*, 2020

```bibtex
@incollection{wenger2020problinsolve,
  author        = {Jonathan Wenger and Philipp Hennig},
  title         = {Probabilistic Linear Solvers for Machine Learning},
  booktitle 	= {Advances in Neural Information Processing Systems (NeurIPS)},
  year          = {2020},
  keywords      = {probabilistic numerics, numerical linear algebra, machine learning},
  url           = {https://github.com/JonathanWenger/probabilistic-linear-solvers-for-ml}
}
```

## Acknowledgements

The authors gratefully acknowledge financial support by the European Research Council through ERC StG Action 757275 / PANAMA; the DFG Cluster of Excellence "Machine Learning - New Perspectives for Science", EXC 2064/1, project number 390727645; the German Federal Ministry of Education and Research (BMBF) through the Tübingen AI Center (FKZ: 01IS18039A); and funds from the Ministry of Science, Research and Arts of the State of Baden-Württemberg.

JW is grateful to the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for support.
