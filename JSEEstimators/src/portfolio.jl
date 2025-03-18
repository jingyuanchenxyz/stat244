"""
Computes the minimum variance portfolio weights.

Param:
* Σ: The covariance matrix
* use_cuda: if gpu 

Return:
* vector of portfolio weights that minimize variance
"""
function min_variance_portfolio(Σ::Matrix{T}; use_cuda=has_cuda) where T <: AbstractFloat
    if use_cuda && has_cuda
        return min_variance_portfolio_cuda(Σ)
    else
        p = size(Σ, 1)
        ones_vec = ones(p)
        
        # Eq. 25
        # subject to w'1 = 1
        Σ_inv = inv(Σ)
        w = (Σ_inv * ones_vec) ./ (ones_vec' * Σ_inv * ones_vec)
        
        return w
    end
end