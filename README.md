
# LatticeFlow.jl

Normalizing-flow methods for improving signal-to-noise in lattice field theory.

![Training demonstration](images/VarianceReduction.gif)



---

## Overview

This project investigates the use of normalizing flows and stochastic automatic
differentiation (SAD) to improve the signal-to-noise ratio of observables in
lattice field theory.

The current implementation focuses on two-dimensional lattice $\phi^4$ theory.

The central idea is to learn an invertible transformation

$$
\phi \longrightarrow \phi' = \phi + f_\theta(\phi),
$$

such that the transformed configurations provide more efficient estimates of
the desired observables.

---

## Physical Model

We consider the lattice $\phi^4$ action

$$
S[\phi]=\sum_x\left[-2\kappa\,\phi_x\sum_{\mu}\phi_{x+\hat\mu}+\phi_x^2+\lambda(\phi_x^2-1)^2\right].
$$


### Lattice geometry

For a two-dimensional lattice,

$$
\phi \in \mathbb{R}^{L_t \times L_x}.
$$

The implementation uses a `Grid` structure to represent the lattice
geometry and its dimensions.

---

## Method

The transformation is parameterized by a neural network,

$$
f_\theta : \phi \mapsto f_\theta(\phi),
$$

and the transformed field is constructed from this learned deformation.

![Residual architecture](images/residual.png)

The training objective is based on a KL-divergence-related loss,

$$
\mathcal{L}(\theta)D_{\mathrm{KL}}\left(q_\theta \,\|\, p\right),
$$

where $p$ denotes the target distribution and $q_\theta$ the distribution
induced by the learned transformation.



### Reweighting

Observables are evaluated using reweighting factors of the form

$$w(\phi)=\exp\left[S(\phi)-S(\phi')+\Delta J+\Delta \log J_f\right],
$$

where the individual terms depend on the particular transformation and
observable.

---

## Repository Structure

```text
LatticeFlow.jl/
│
├── src/
│   ├── action.jl
│   ├── model.jl
│   ├── loss.jl
│   ├── training.jl
│   ├── data.jl
│   ├── observables.jl
│   └── ...
│
├── scripts/
│   ├── train.jl
│   └── plot_results.jl
│
├── test/
│   ├── test_action.jl
│   ├── test_model.jl
│   └── ...
│
├── priors/
├── results/
├── Project.toml
└── README.md