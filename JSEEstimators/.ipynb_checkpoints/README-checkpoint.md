# JSEEstimators.jl

A Julia package for James-Stein Eigenvector (JSE) estimators for covariance matrices in high-dimensional regimes. Based off:
L.R. Goldberg,& A.N. Kercheval,  James–Stein for the leading eigenvector, Proc. Natl. Acad. Sci. U.S.A. 120 (2) e2207046120, https://doi.org/10.1073/pnas.2207046120 (2023).
and work done at Berkeley CDAR.

## Features

- Implementation of JSE estimator for the leading eigenvector of a factor-based covariance matrix
- Support for both CPU and GPU (CUDA) implementations
- Minimum variance portfolio optimization
- Performance metrics for comparing estimators
- Simulation utilities for factor model data
- Plotting functions for visualizing results

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jingyuanchenxyz/stat244/JSEEstimators.jl")
```

## Example Usage

```julia
using JSEEstimators

# Simulate factor data
n, p = 100, 500
X, β, Σ = simulate_factor_data(n, p)
S = cov(X, dims=2)

# Compute JSE estimator
h_jse = jse_estimator(S, n)

# Estimate covariance matrix using the JSE estimator
Σ_jse = factor_cov_matrix(S, n, method=:jse)

# Compute minimum variance portfolio
w = min_variance_portfolio(Σ_jse)

# Run trials to compare estimators
metrics = run_hl_trials(n, p, 10)

# Plot results
plot_hl_results(metrics)
```

## GPU Acceleration

JSEEstimators.jl supports GPU acceleration via CUDA.jl. If CUDA.jl is installed and a compatible GPU is detected, JSEEstimators will automatically use GPU acceleration for computationally intensive operations.

## Functions

### Core Functions
- `jse_estimator`: Computes the James-Stein Eigenvector estimator
- `factor_cov_matrix`: Estimates a factor-based covariance matrix

### Portfolio Optimization
- `min_variance_portfolio`: Computes minimum variance portfolio weights

### Metrics
- `angular_distance`: Computes the angular distance between vectors
- `optimization_bias`: Measures the distance from true optimal weights
- `variance_forecast_ratio`: Measures the accuracy of variance forecasts
- `tracking_error`: Computes tracking error between portfolios

### Simulation
- `simulate_factor_data`: Generates synthetic factor data
- `run_hl_trials`: Runs simulation trials to compare estimators

### Other
- `has_cuda_support`: Checks if CUDA is available