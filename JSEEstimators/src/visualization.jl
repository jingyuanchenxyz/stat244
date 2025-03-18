"""
Plots comparing the performance of different estimators for a given metric.

Params:
* metrics: The metrics dict from run_hl_trials
* metric_key: The key of the metric to plot
* title: you know what
"""
function plot_metrics(metrics::Dict, metric_key::Symbol, title::String="")
    boxplot(
        [metrics[metric_key][:raw], metrics[metric_key][:pca], metrics[metric_key][:jse]],
        labels=["Raw" "PCA" "JSE"],
        title=isempty(title) ? string(metric_key) : title,
        ylabel=string(metric_key),
        yformatter = :auto
    )
end

"""
Plots
Params:
* metrics: The metrics dict from run_hl_trials

Return:
* A plot object
"""
function plot_hl_results(metrics::Dict)
    p = plot(layout=(2,2), size=(1200, 800), margin=5mm)
    
    subplot_titles = [
        "Angular Distance (radians)", 
        "Optimization Bias (smaller is better)",
        "Variance Forecast Ratio (closer to 1 is better)",
        "Tracking Error (smaller is better)"
    ]
    
    metric_keys = [:angular, :optim_bias, :vfr, :tracking_error]
    
    for (i, (key, title)) in enumerate(zip(metric_keys, subplot_titles))
        if haskey(metrics, key) && 
           all(haskey(metrics[key], method) && !isempty(metrics[key][method]) 
               for method in [:raw, :pca, :jse])
            
            boxplot!(
                p[i],
                [metrics[key][:raw], metrics[key][:pca], metrics[key][:jse]],
                labels=["Raw" "PCA" "JSE"],
                title=title,
                titlefontsize=10,
                xtickfontsize=8,
                ytickfontsize=8,
                yformatter=:auto
            )
        else
            annotate!(p[i], 0.5, 0.5, text("No data available for $(key)", 10))
        end
    end
    
    return p
end

"""
Benchmark different implementations of JSE estimator.

Params:
* n: num of obs
* p: num of vars
* compare_cuda: benchmark CUDA versions or not
"""
function benchmark_implementations(n::Int, p::Int; compare_cuda=has_cuda)
    X, β, Σ = simulate_factor_data(n, p)
    S = cov(X, dims=2)
    
    println("Benchmarking CPU implementations:")
    println("JSE estimator with Symmetric matrix:")
    @btime jse_estimator($S, $n, use_symmetric=true, use_cuda=false)
    
    println("\nJSE estimator without Symmetric matrix:")
    @btime jse_estimator($S, $n, use_symmetric=false, use_cuda=false)
    
    println("\nFactor covariance matrix with raw method:")
    @btime factor_cov_matrix($S, $n, method=:raw, use_cuda=false)
    
    println("\nFactor covariance matrix with JSE method:")
    @btime factor_cov_matrix($S, $n, method=:jse, use_cuda=false)
    
    println("\nMinimum variance portfolio:")
    @btime min_variance_portfolio($Σ, use_cuda=false)
    
    if compare_cuda && has_cuda
        println("\n\nBenchmarking CUDA implementations:")
        println("JSE estimator with CUDA:")
        @btime jse_estimator($S, $n, use_cuda=true)
        
        println("\nFactor covariance matrix with raw method (CUDA):")
        @btime factor_cov_matrix($S, $n, method=:raw, use_cuda=true)
        
        println("\nFactor covariance matrix with JSE method (CUDA):")
        @btime factor_cov_matrix($S, $n, method=:jse, use_cuda=true)
        
        println("\nMinimum variance portfolio (CUDA):")
        @btime min_variance_portfolio($Σ, use_cuda=true)
    elseif compare_cuda
        println("\nCUDA benchmarks skipped: CUDA not available")
    end
end

"""
CPU vs GPU comparison for different problem sizes.

Params:
* sizes: array of problem dimensions to test
* trials: num of trials for each size
"""
function benchmark_cuda_vs_cpu(sizes=[100, 200, 500, 1000], trials=5)
    if !has_cuda
        println("CUDA not available, skipping comparison")
        return nothing
    end
    
    results = Dict(
        :cpu_time => Float64[],
        :gpu_time => Float64[],
        :size => Int[]
    )
    
    for p in sizes
        push!(results[:size], p)
        
        # Warmup
        run_hl_trials(40, p, 1, use_cuda=false)
        run_hl_trials(40, p, 1, use_cuda=true)
        
        # CPU timing
        cpu_start = time()
        run_hl_trials(40, p, trials, use_cuda=false)
        cpu_time = time() - cpu_start
        push!(results[:cpu_time], cpu_time)
        
        # GPU timing
        gpu_start = time()
        run_hl_trials(40, p, trials, use_cuda=true)
        gpu_time = time() - gpu_start
        push!(results[:gpu_time], gpu_time)
        
        println("Size p=$p: CPU time = $(round(cpu_time, digits=2))s, GPU time = $(round(gpu_time, digits=2))s, Speedup = $(round(cpu_time/gpu_time, digits=2))x")
    end
    
    # Create comparison plot
    p = plot(results[:size], results[:cpu_time], 
        label="CPU", marker=:circle, legend=:topleft,
        xlabel="Dimension (p)", ylabel="Runtime (seconds)",
        title="CPU vs GPU Performance Comparison ($trials trials)")
    plot!(p, results[:size], results[:gpu_time], 
        label="GPU", marker=:square)
    
    # Plot speedup on secondary axis
    speedup = results[:cpu_time] ./ results[:gpu_time]
    plot!(twinx(p), results[:size], speedup, 
        label="Speedup", color=:green, marker=:diamond,
        ylabel="Speedup (x)", legend=:topright)
    
    display(p)
    return results
end