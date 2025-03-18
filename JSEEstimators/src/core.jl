"""
Computes the James-Stein Eigenvector (JSE) estimator for the leading eigenvector
of a factor-based covariance matrix estimated from the high-dimensional limit regime.

Params:
* S: sample covariance matrix
* n: num of obs used to compute S
* use_symmetric: Symmetric wrapper for better performance
* use_cuda: use GPU acceleration if available

Return:
* normalized JSE estimator of the leading eigenvector
"""
function jse_estimator(S::Matrix{T}, n::Int; use_symmetric=true, use_cuda=has_cuda) where T <: AbstractFloat
    if use_cuda && has_cuda
        return jse_estimator_cuda(S, n)
    else
        p = size(S, 1)
        
        F = use_symmetric ? eigen(Symmetric(S)) : eigen(S)
        
        #Note to self: largest eigenvalue is last in Julia
        λ² = F.values[end]
        h = F.vectors[:, end]
        
        if sum(h) < 0
            h = -h
        end
        
        # Eq 9
        ν² = (tr(S) - λ²) / (p * (n - 1))
        
        # Eq 8
        λ = sqrt(λ²)
        m_h = mean(h)
        s² = sum((λ*h .- λ*m_h).^2)/p
        
        eps_threshold = eps(T) * λ²
        if s² <= eps_threshold
            return h ./ norm(h)  # 0 shrinkage needed
        end
        
        # Eq 7
        c_JSE = clamp(1 - ν²/s², zero(T), one(T))
        
        # Eq 6
        h_JSE = m_h .+ c_JSE.*(h .- m_h)
        return h_JSE ./ norm(h_JSE)
    end
end

"""
Factor based covariance matrix estimator using either raw or JSE eigenvector (Eq 37)

Params:
* S: sample covariance matrix
* n: num of obs used to compute S
* method: :raw, :jse, or :pca for different eigenvector estimators
* use_cuda: use GPU acceleration if available

Return:
* estimated covariance matrix
"""
function factor_cov_matrix(S::Matrix{T}, n::Int; method=:jse, use_cuda=has_cuda) where T <: AbstractFloat
    if use_cuda && has_cuda
        return factor_cov_matrix_cuda(S, n, method=method)
    else
        p = size(S, 1)
        F = eigen(Symmetric(S))
        λ² = F.values[end]
        h = F.vectors[:, end]
        
        # Eq. 31
        ℓ² = (tr(S) - λ²) / (n - 1)
        
        if method == :raw
            b = h
        elseif method == :jse
            b = jse_estimator(S, n, use_cuda=false)
        elseif method == :pca
            # modified PCA: correct eigenvalue but keep sample eigenvector
            b = h
        else
            error("Unknown method: $method")
        end
        
        # factor covariance matrix Eq. 37
        Σ_hat = (λ² - ℓ²) * (b * b') + (n/p) * ℓ² * I
        
        return Σ_hat
    end
end