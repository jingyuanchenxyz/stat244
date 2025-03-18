using JSEEstimators
using Test
using LinearAlgebra
using Statistics

@testset "JSEEstimators.jl" begin
    # Core functionality tests
    @testset "Core Estimators" begin
        n, p = 40, 100
        X, β, Σ = simulate_factor_data(n, p, seed=42)
        S = cov(X, dims=2)
        
        h_jse = jse_estimator(S, n, use_cuda=false)
        @test length(h_jse) == p
        @test norm(h_jse) ≈ 1.0 atol=1e-10
        
        Σ_jse = factor_cov_matrix(S, n, method=:jse, use_cuda=false)
        @test size(Σ_jse) == (p, p)
        @test isapprox(Σ_jse, Σ_jse', atol=1e-10)  # Symmetry
    end
    
    @testset "Portfolio Optimization" begin
        n, p = 40, 100
        X, β, Σ = simulate_factor_data(n, p, seed=42)
        
        w = min_variance_portfolio(Σ, use_cuda=false)
        @test length(w) == p
        @test sum(w) ≈ 1.0 atol=1e-10
    end
    
    @testset "Metrics" begin
        n, p = 40, 100
        X, β, Σ = simulate_factor_data(n, p, seed=42)
        S = cov(X, dims=2)
        
        Σ_jse = factor_cov_matrix(S, n, method=:jse, use_cuda=false)
        h_jse = jse_estimator(S, n, use_cuda=false)
        w_true = min_variance_portfolio(Σ, use_cuda=false)
        w_est = min_variance_portfolio(Σ_jse, use_cuda=false)
        
        # Angular distance
        ad = angular_distance(h_jse, β ./ norm(β))
        @test 0 <= ad <= π
        
        # Optimization bias
        ob = optimization_bias(w_est, w_true, Σ)
        @test ob >= 0
        
        # Variance forecast ratio
        vfr = variance_forecast_ratio(w_est, Σ_jse, Σ)
        @test vfr > 0
        
        # Tracking error
        te = tracking_error(w_est, w_true, Σ)
        @test te >= 0
    end
    
    # Run integrated performance test
    @testset "Integration" begin
        metrics = run_hl_trials(40, 100, 2, use_cuda=false)
        @test haskey(metrics, :angular)
        @test haskey(metrics, :optim_bias)
        @test haskey(metrics, :vfr)
        @test haskey(metrics, :tracking_error)
    end
end