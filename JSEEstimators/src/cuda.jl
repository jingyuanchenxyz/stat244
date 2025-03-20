"""
CUDA JSE estimator

Params:
* S: sample covariance matrix 
* n: num of obs used to compute S

Return:
* normalized JSE estimator of the leading eigenvector
"""
function jse_estimator_cuda(S::Matrix{T}, n::Int) where T <: AbstractFloat
    if !has_cuda
        error("CUDA not available, use CPU implementation instead")
    end
    
    p = size(S, 1)
    
    # move to GPU
    d_S = CuMatrix(Symmetric(S))
    
    # eigendecomposition on GPU
    values, vectors = CUDA.CUSOLVER.syevd!('V', 'U', d_S)
    
    # batch transfer to CPU to avoid scalar indexing on GPU arr
    cpu_values = Array(values)
    cpu_vectors = Array(vectors)
    
    λ² = cpu_values[end]
    h = cpu_vectors[:, end]
    
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

"""
CUDA factor_cov_matrix

Params:
* S: sample covariance matrix
* n: num of obs used to compute S
* method: :raw, :jse, or :pca for different eigenvector estimators

Return:
* estimated covariance matrix
"""
function factor_cov_matrix_cuda(S::Matrix{T}, n::Int; method=:jse) where T <: AbstractFloat
    if !has_cuda
        error("CUDA not avail, using CPU instead")
    end
    
    p = size(S, 1)
    
    d_S = CuMatrix(Symmetric(S))
    
    values, vectors = CUDA.CUSOLVER.syevd!('V', 'U', d_S)
    
    # back to CPU to avoid scalar indexing
    cpu_values = Array(values)
    cpu_vectors = Array(vectors)
    
    λ² = cpu_values[end]
    h = cpu_vectors[:, end]
    
    # Eq. 31
    ℓ² = (tr(S) - λ²) / (n - 1)
    
    if method == :raw
        b = h
    elseif method == :jse
        b = jse_estimator_cuda(S, n)
    elseif method == :pca
        b = h
    else
        error("Unknown method: $method")
    end
    
    # b to GPU
    d_b = CuVector(b)
    
    d_bb_t = d_b * d_b'
    
    # Create identity matrix on CPU and transfer to GPU
    # This avoids scalar indexing when adding the identity matrix
    eye_cpu = Matrix{T}(I, p, p)
    d_eye = CuMatrix(eye_cpu)
    d_Σ_hat = (λ² - ℓ²) * d_bb_t + (n/p) * ℓ² * d_eye
    
    return Array(d_Σ_hat)
end

"""
CUDA min_variance_portfolio

Params:
* Σ: The covariance matrix

Return:
* vector of portfolio weights that minimize variance
"""
function min_variance_portfolio_cuda(Σ::Matrix{T}) where T <: AbstractFloat
    if !has_cuda
        error("CUDA not avail, using CPU instead")
    end
    
    p = size(Σ, 1)
    
    d_Σ = CuMatrix(Σ)
    d_ones = CUDA.ones(p)
    
    # inverse on GPU
    d_Σ_inv = CUDA.inv(d_Σ)
    
    d_numerator = d_Σ_inv * d_ones
    denominator = d_ones' * d_numerator
    d_w = d_numerator ./ denominator
    
    return Array(d_w)
end