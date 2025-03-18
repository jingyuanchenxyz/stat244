"""
The angular distance (smallest angle) between two vectors

Params:
* v1, v2: The vectors to compare

Return:
* The angle between the vectors in radians
"""
function angular_distance(v1::Vector{T}, v2::Vector{T}) where T <: AbstractFloat
    v1_norm = v1 ./ norm(v1)
    v2_norm = v2 ./ norm(v2)
    
    if sum(v1_norm) < 0
        v1_norm = -v1_norm
    end
    if sum(v2_norm) < 0
        v2_norm = -v2_norm
    end
    
    cos_sim = clamp(dot(v1_norm, v2_norm), -1.0, 1.0)
    
    return acos(cos_sim)
end

"""
Compute the optimization bias metric, which measures how far the estimated 
portfolio weights are from the true optimal weights.

Params:
* w_est: The estimated portfolio weights
* w_true: The true optimal portfolio weights
* Σ: The true covariance matrix

Return:
* The optimization bias metric
"""
function optimization_bias(w_est::Vector{T}, w_true::Vector{T}, Σ::Matrix{T}) where T <: AbstractFloat
    # Eq. 27
    tvr = (w_true' * Σ * w_true) / (w_est' * Σ * w_est)
    
    return 1.0 - tvr
end

"""
Compute the variance forecast ratio which measures the accuracy of the variance forecast

Params:
* w: portfolio weights
* Σ_est: estimated covariance matrix
* Σ_true: true covariance matrix

Return:
* variance forecast ratio (1.0 is perfect, closer the better)
"""
function variance_forecast_ratio(w::Vector{T}, Σ_est::Matrix{T}, Σ_true::Matrix{T}) where T <: AbstractFloat
    # Eq. 26
    vfr = (w' * Σ_est * w) / (w' * Σ_true * w)
    
    return vfr
end

"""
Tracking error between the estimated and true optimal portfolios.

Params:
* w_est: estimated portfolio weights
* w_true: true optimal portfolio weights
* Σ: true covariance matrix

Return:
* The tracking error
"""
function tracking_error(w_est::Vector{T}, w_true::Vector{T}, Σ::Matrix{T}) where T <: AbstractFloat
    # Eq. 28
    te_squared = (w_est - w_true)' * Σ * (w_est - w_true)
    
    return sqrt(te_squared)
end