using LinearAlgebra
using QuadGK

struct Gaussian{T}
    mean
    covariance::Matrix{T}
end

# function Gaussian(mean, covariance::Matrix{T}) where {T}
#     @assert length(mean) == size(covariance, 1) "mean and covariance size mismatch"
#     @assert size(covariance, 1) == size(covariance, 2) "covariance matrix must be square"
#     Gaussian{T}(mean, Matrix(covariance))
# end

""" 
    from_moment(μ, P) 
    Construct a Gaussian distribution with given mean and covariance matrix.

    # Arguments 
    - `μ` The mean vector of the Gaussian distribution.
    - `P` The covariance matrix of the Gaussian distribution.
"""
function from_moment(μ::Vector, P)
    return Gaussian(μ, Matrix(P))
end 

""" 
    from_sqrt_moment(μ, S) 
    Construct a Gaussian distribution with given mean and square root covariance matrix.

    # Arguments 
    - `μ` The mean vector of the Gaussian distribution.
    - `S` The square root covariance matrix (upper triangular) of the Gaussian distribution.

"""
function from_sqrt_moment(μ::Vector, S)
    @assert istriu(Matrix(S)) "S must be upper triangular"
    return Gaussian(μ, Matrix(S))
end 

""" 
    from_info(η, Λ) 
    Construct a Gaussian distribution with given information vector and information matrix.

    # Arguments 
    - `η` The information vector of the Gaussian distribution.
    - `Λ` The information matrix of the Gaussian distribution.

"""
function from_info(η::Vector, Λ)
    return Gaussian(η, Matrix(Λ))
end 

""" 
    from_sqrt_info(ν, Ξ) 
    Construct a Gaussian distribution with given information vector and square root information matrix.

    # Arguments 
    - `ν` The information vector of the Gaussian distribution.
    - `Ξ` The square root information matrix (upper triangular) of the Gaussian distribution.

"""
function from_sqrt_info(ν::Vector, Ξ)
    @assert istriu(Matrix(Ξ)) "Ξ must be upper triangular"
    return Gaussian(ν, Matrix(Ξ))
end 


"""
    log_sqrt_pdf(x, pdf) 

Compute the logarithm of a multivariate normal distribution in full covariance or square-root form at the value `x`. 

# Arguments
- `x` The input vector at which to evaluate the log-likelihood.
- `pdf` A multivariate normal distribution with mean `μ` and covariance matrix `Σ` or square-root covariance matrix `S` such that SᵀS = Σ.

# Returns
- The log of the probability distribution function evaluated at `x`.
"""
function log_pdf(x, pdf::Gaussian; grad=grad, sqrt=sqrt)
    
    μ = pdf.mean
    @assert length(x) == length(μ) "Input x and mean μ must have same length"
    n = length(x)

    Δ = x .- μ  

    if sqrt
        
        S = pdf.covariance
    
        @assert istriu(S) "S is not upper triangular"
        
        w = LowerTriangular(transpose(S)) \ Δ   
        logpdf = -(n/2)*log(2π)-sum(log.(abs.(diag(Matrix(S)))))-(1/2)*dot(w,w)

        if grad
            gradient = -UpperTriangular(S) \ w      # Gradient ∇logp = -S⁻¹ * w
            return logpdf, gradient
        else 
            return logpdf                           # Return log N(x; μ, S)
        end 
    else 
        Σ = pdf.covariance
        logpdf = -(n/2)*log(2π)-(1/2)*logdet(Σ)-(1/2)*dot(Δ,Σ\Δ)

        if grad
            gradient = -Σ\Δ
            return logpdf, gradient
        else 
            return logpdf
        end 
    end 
end 


"""
    conditional(density::Gaussian, idx_𝑥::Int, idx_𝑦::Int, 𝑦)

Given the joint Gaussian N(μ, Σ) and index sets for variables A and B, return 𝑝(𝑥 | 𝑦).

# Arguments
- `density::Gaussian`: The joint probability distribution function as a Gaussian density 𝑝(𝑥, 𝑦).
- `idx_𝑥::Vector`: The indices of the variables to condition on.
- `idx_𝑦::Vector`: The indices of the variables to condition on.
- `𝑦::Vector`: The values of the variables to condition on.

# Returns
- `μ_cond::Vector`: Conditional mean μ_A|B
- `Σ_cond::Matrix`: Conditional covariance Σ_A|B
"""
function conditional(density, idx_𝑥, idx_𝑦, 𝑦; sqrt=false)

    if sqrt
        # The conditional distribution of 𝑦 given 𝑥 is given by 𝑝(𝑦 | 𝑥) = 𝑁(μ𝑦 + S₂ᵀS₁⁻ᵀ(𝑥 - μ𝑥), S₃)
        μ = density.mean
        S = density.covariance 

        # The joint distribution passed to this function must be in the form p([𝑦; 𝑥]) and not p([𝑥; 𝑦]) 
        # Extract the blocks S₁, S₂, S₃ from S, this assumes that the square-root covariance is stored as S and not Sᵀ
        S₁ = S[idx_𝑥, idx_𝑥]
        S₂ = S[idx_𝑥, idx_𝑦]
        S₃ = S[idx_𝑦, idx_𝑦]

        # Compute S₁⁻ᵀ(𝑥 - μ𝑥) by solving the linear system S₁ * w = 𝑦 - μ𝑥
        w = S₁ \ (𝑦 - μ[idx_𝑥])

        # Compute the conditional mean μ_cond = μ𝑦 + S₂ᵀS₁⁻ᵀ(𝑥 - μ𝑥)
        μ_cond = μ[idx_𝑦] + S₂' * w

        # Compute the conditional square-root covariance S_cond = S₃, that is the square-root covariance of p(𝑦 | 𝑥)
        S_cond = S₃

        return from_sqrt_moment(μ_cond, S_cond)

    else

        μ = density.mean
        Σ = density.covariance
        
        μ𝑥 = μ[idx_𝑥]
        μ𝑦 = μ[idx_𝑦]

        Σ𝑥𝑥 = Σ[idx_𝑥, idx_𝑥]
        Σ𝑥𝑦 = Σ[idx_𝑥, idx_𝑦]
        Σ𝑦𝑥 = Σ[idx_𝑦, idx_𝑥]
        Σ𝑦𝑦 = Σ[idx_𝑦, idx_𝑦]

        # Compute the new mean and covariance of the conditional distribution 𝑝(𝑥 | 𝑦)
        # Dont invert the matrix (Σ𝑦𝑦⁻¹) - https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/

        # Instead, solve the linear system Σ𝑦𝑦 * w = v to find w = Σ𝑦𝑦⁻¹ * v 
        w = Σ𝑦𝑦 \ (𝑦 - μ𝑦)

        # Compute the conditional mean μ𝑥|𝑦 = μ𝑥 + Σ𝑥𝑦 * Σ𝑦𝑦⁻¹ * (𝑦 - μ𝑦)
        μ_cond = μ𝑥 + Σ𝑥𝑦 * w  

        # Again solve the linear system Σ𝑦𝑦 * w = Σ𝑦𝑥 to find w = Σ𝑦𝑦⁻¹ * Σ𝑦𝑥
        w = Σ𝑦𝑦 \ Σ𝑦𝑥

        # Compute the conditional covariance Σ𝑥|𝑦 = Σ𝑥𝑥 - Σ𝑥𝑦 * Σ𝑦𝑦⁻¹ * Σ𝑦𝑥
        Σ_cond = Σ𝑥𝑥 - Σ𝑥𝑦 * w  

        # Return the conditional distribution 𝑝(𝑥 | 𝑦)
        return from_moment(μ_cond, Σ_cond)
    end 
end 


"""
    marginal(density::Gaussian, idx::Vector{Int})

Compute the marginal distribution of a Gaussian density over a subset of variables.

# Arguments
- `density::Gaussian`: The Gaussian density to marginalize.
- `idx::Vector{Int}`: The indices of the variables to marginalise over.
"""
function marginal(density::Gaussian, idx::Vector{Int}; sqrt=sqrt)
    if sqrt
        #            Σ = SᵀS

        # [ Σ𝑥𝑥  Σ𝑥𝑦 ] = [  S₁  S₂ ]ᵀ [ S₁  S₂ ]
        # [ Σ𝑦𝑥  Σ𝑦𝑦 ] = [  0   S₃ ]  [ 0   S₃ ]

        #              = [ S₁ᵀ  0  ]  [ S₁  S₂ ]
        #              = [ S₂ᵀ  S₃ᵀ]  [ 0   S₃ ]

        #              = [ S₁ᵀS₁      S₁ᵀS₂     ]
        #              = [ S₂ᵀS₁  S₂ᵀS₂ + S₃ᵀS₃ ]

        S𝑥𝑥 = density.covariance
        S₂ = S𝑥𝑥[idx, idx:end]
        S₃ = S𝑥𝑥[idx:end, idx:end]

        R₁ = qr([S₂; S₃]).R

        return from_sqrt_moment(density.mean[idx], R₁)

    else 
        return from_moment(density.mean[idx], density.covariance[idx, idx])
    end 
end


"""
join(density_𝑥, density_𝑦; sqrt=false)

Construct the joint distribution of two independent Gaussian densities.

# Arguments
- `density_𝑥`: A Gaussian distribution representing the first random variable.
- `density_𝑦`: A Gaussian distribution representing the second random variable.

# Keyword Arguments
- `sqrt`: If `true`, constructs the joint in square-root form (not yet implemented). Defaults to `false`.

# Returns
- A new Gaussian representing the joint distribution, with concatenated means and a block-diagonal covariance matrix.
"""
function join(density_𝑥, density_𝑦; sqrt=false)

    μ = vcat(density_𝑥.mean, density_𝑦.mean)
   
    n𝑥 = size(density_𝑥.covariance, 1)
    n𝑦 = size(density_𝑦.covariance, 1)

    if sqrt
        error("Not implemented yet") # TODO
        return from_sqrt_moment(μ, S)
    else 
        # Create block diagonal matrix from the two covariance matrices
        Σ = zeros(n𝑥 + n𝑦, n𝑥 + n𝑦)
        Σ[1:n𝑥, 1:n𝑥] = density_𝑥.covariance
        Σ[n𝑥+1:end, n𝑥+1:end] = density_𝑦.covariance
        return from_moment(μ, Σ)
    end 
end


"""
    sum(𝐗, 𝐘)

Compute the sum of two independent random variables (not necessarily Gaussian) 𝐗 + 𝐘 by computing z = (f ∗ g), that is the convolution of the two probability density functions f and g.

# Arguments
- `f`: A probability density function representing the random variable 𝐗.
- `g`: A probability density function representing the random variable 𝐘. 

# Returns
- `z`: A probability density function that represents 𝐙, that is the sum of the two random variables 𝐗 + 𝐘.

"""
function sum(f::Function, g::Function)
    
    # Compute the function z(s) = ∫ f(x) g(z - x) dx

    function z(s)
        
        # Define the integrand, that is the function to be integrated
        integrand = (x) -> f(x) * g(s - x)

        # Integrate the integrand
        value, error = quadgk(integrand, -Inf, Inf)

        return value
    end 

    return z
end 


function add(𝑥₁::Gaussian, 𝑥₂::Gaussian; sqrt=sqrt)
    
    @assert length(𝑥₁.mean) == length(𝑥₂.mean) "mean length mismatch"
    @assert size(𝑥₁.covariance) == size(𝑥₂.covariance) "covariance size mismatch"

    μ = 𝑥₁.mean + 𝑥₂.mean

    if sqrt
        
        # Prepare the matrix for QR decomposition
        A = vcat(𝑥₁.covariance, 𝑥₂.covariance)

        # Perform QR decomposition
        F = qr(A)

        # Extract the upper triangular matrix R, by default the QR decomposition returns the upper square non-zero part of the matrix
        S = F.R   

        return from_sqrt_moment(μ, S)
    else 
        
        Σ = 𝑥₁.covariance + 𝑥₂.covariance
        return from_moment(μ, Σ)
    end 
end 