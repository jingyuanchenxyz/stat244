"""
Simulate data from a one factor model where β ~ N(1, 0.25)
factor returns with σ=0.16, specific returns with δ=0.60

Params:
* n: Num of obs
* p: Numb of vars
* σ: SD of factor returns (default: 0.16)
* δ: SD of specific returns (default: 0.60)
* mean_beta: mean of the beta distribution (default: 1.0)
* var_beta: var of the beta distribution (default: 0.25)

Return:
* X:  synthetic p x n data matrix
* β: true factor loadings
* Σ: true covariance matrix
"""
function simulate_factor_data(n::Int, p::Int; σ=0.16, δ=0.60, mean_beta=1.0, var_beta=0.25, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    β = rand(Normal(mean_beta, sqrt(var_beta)), p)
    
    # (common factor f)
    f = rand(Normal(0, σ), n)
    
    # specific returns (with t-distribution with 5)
    ϵ = rand(TDist(5), (p, n)) .* (δ / sqrt(5/(5-2)))
    
    X = β * f' + ϵ
    
    # Eq. 11
    Σ = (σ^2) * (β * β') + (δ^2) * I
    
    return X, β, Σ
end

"""
HL trials to compare different estimators.

Params:
* n: num of obs
* p: num of vars
* num_trials: num of sim trials
* use_cuda: use GPU acceleration if available
* kwargs: params passed to simulate_factor_data

Return:
* A dict with preformance metrics
"""
function run_hl_trials(n=40, p=500, num_trials=100; use_cuda=has_cuda, kwargs...)
    metrics = Dict(
        :angular => Dict(:raw => Float64[], :pca => Float64[], :jse => Float64[]),
        :optim_bias => Dict(:raw => Float64[], :pca => Float64[], :jse => Float64[]),
        :vfr => Dict(:raw => Float64[], :pca => Float64[], :jse => Float64[]),
        :tracking_error => Dict(:raw => Float64[], :pca => Float64[], :jse => Float64[])
    )
    
    for trial in 1:num_trials
        X, β, Σ = simulate_factor_data(n, p, seed=trial; kwargs...)
        S = cov(X, dims=2)
        w_true = min_variance_portfolio(Σ, use_cuda=use_cuda)
        
        if use_cuda && has_cuda
            d_S = CuMatrix(Symmetric(S))
            F = CUDA.CUSOLVER.syevd!('V', 'U', d_S)
            h_raw = Array(F.vectors[:, end])
        else
            h_raw = eigen(Symmetric(S)).vectors[:, end]
        end
        
        if sum(h_raw) < 0
            h_raw = -h_raw
        end
        
        h_jse = jse_estimator(S, n, use_cuda=use_cuda)
        
        β_norm = β ./ norm(β)
        if sum(β_norm) < 0
            β_norm = -β_norm
        end
        
        push!(metrics[:angular][:raw], angular_distance(h_raw, β_norm))
        push!(metrics[:angular][:pca], angular_distance(h_raw, β_norm))
        push!(metrics[:angular][:jse], angular_distance(h_jse, β_norm))
        
        for method in [:raw, :pca, :jse]
            Σ_est = factor_cov_matrix(S, n, method=method, use_cuda=use_cuda)
            w_est = min_variance_portfolio(Σ_est, use_cuda=use_cuda)
            
            push!(metrics[:optim_bias][method], optimization_bias(w_est, w_true, Σ))
            push!(metrics[:vfr][method], variance_forecast_ratio(w_est, Σ_est, Σ))
            push!(metrics[:tracking_error][method], tracking_error(w_est, w_true, Σ))
        end
    end
    
    return metrics
end

function run_hl_trials_verbose(n=40, p=500, num_trials=5; use_cuda=has_cuda, kwargs...)
    if use_cuda && has_cuda
        println("Running with CUDA")
    else
        println("Running with CPU")
    end
    
    metrics = run_hl_trials(n, p, num_trials, use_cuda=use_cuda; kwargs...)
    
    println("=== Mean Performance Metrics ($num_trials trials) ===")
    println("Angular Distance (radians, smaller is better):")
    println("  Raw: $(round(mean(metrics[:angular][:raw]), digits=4))")
    println("  PCA: $(round(mean(metrics[:angular][:pca]), digits=4))")
    println("  JSE: $(round(mean(metrics[:angular][:jse]), digits=4))")
    println("  Improvement: $(round(mean(metrics[:angular][:raw]) - mean(metrics[:angular][:jse]), digits=4))")
    
    println("\nOptimization Bias (smaller is better):")
    println("  Raw: $(round(mean(metrics[:optim_bias][:raw]), digits=4))")
    println("  PCA: $(round(mean(metrics[:optim_bias][:pca]), digits=4))")
    println("  JSE: $(round(mean(metrics[:optim_bias][:jse]), digits=4))")
    println("  Raw-JSE Improvement: $(round(mean(metrics[:optim_bias][:raw]) - mean(metrics[:optim_bias][:jse]), digits=4))")
    println("  PCA-JSE Improvement: $(round(mean(metrics[:optim_bias][:pca]) - mean(metrics[:optim_bias][:jse]), digits=4))")
    
    println("\nVariance Forecast Ratio (closer to 1 is better):")
    println("  Raw: $(round(mean(metrics[:vfr][:raw]), digits=4))")
    println("  PCA: $(round(mean(metrics[:vfr][:pca]), digits=4))")
    println("  JSE: $(round(mean(metrics[:vfr][:jse]), digits=4))")
    
    println("\nTracking Error (smaller is better):")
    println("  Raw: $(round(mean(metrics[:tracking_error][:raw]), digits=4))")
    println("  PCA: $(round(mean(metrics[:tracking_error][:pca]), digits=4))")
    println("  JSE: $(round(mean(metrics[:tracking_error][:jse]), digits=4))")
    
    return metrics
end

"""
Plots

Params:
* metrics: The metrics dict from run_hl_trials

Return:
* A plot object
"""
function run_paper_simulation(plot_results=true; n=40, p=500, trials=100, use_cuda=has_cuda)
    println("simulations with (n=$n, p=$p, $trials trials)...")
    metrics = run_hl_trials_verbose(n, p, trials, use_cuda=use_cuda, σ=0.16, δ=0.60, mean_beta=1.0, var_beta=0.25)
    
    if plot_results
        plt = plot_hl_results(metrics)
        display(plt)
    end
    
    return metrics
end

@testset "JSE HL Performance" begin
    # small test
    metrics = run_hl_trials(40, 100, 10, use_cuda=false)
    
    # angular improvement
    @test mean(metrics[:angular][:jse]) < mean(metrics[:angular][:raw])
    
    # optimization bias with tolerance
    @test mean(metrics[:optim_bias][:jse]) < mean(metrics[:optim_bias][:raw]) + 0.02
    
    # variance forecast ratio (should be closer to 1.0)
    avg_raw = mean(metrics[:vfr][:raw])
    avg_jse = mean(metrics[:vfr][:jse])
    
    # distance from 1.0
    dist_raw = abs(avg_raw - 1.0)
    dist_jse = abs(avg_jse - 1.0)
    
    @test dist_jse < dist_raw
    
    # tracking error should be smaller for JSE
    @test mean(metrics[:tracking_error][:jse]) < mean(metrics[:tracking_error][:raw])
end