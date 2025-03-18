module JSEEstimators

using LinearAlgebra
using Statistics
using Distributions
using Test
using Plots
using Random
using Measures
using BenchmarkTools
using StatsPlots

const has_cuda = try
    using CUDA
    CUDA.functional()
catch
    false
end

include("core.jl")
include("cuda.jl")
include("portfolio.jl")
include("metrics.jl")
include("simulation.jl")
include("visualization.jl")

export 
    # Core estimator functions
    jse_estimator,
    factor_cov_matrix,
    
    # Portfolio functions
    min_variance_portfolio,
    
    # Metrics
    angular_distance,
    optimization_bias,
    variance_forecast_ratio,
    tracking_error,
    
    # Simulation
    simulate_factor_data,
    run_hl_trials,
    run_hl_trials_verbose,
    run_paper_simulation,
    
    # Plotting and benchmarks
    plot_metrics,
    plot_hl_results,
    benchmark_implementations,
    benchmark_cuda_vs_cpu,
    
    # Utility
    has_cuda_support

end