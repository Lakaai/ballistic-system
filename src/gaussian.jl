using LinearAlgebra


struct Gaussian{T}
    mean
    covariance::Matrix{T}
end


""" 
    from_moment(μ, P) 
    Construct a Gaussian distribution with given mean and covariance matrix.

    # Arguments 
    - `μ` The mean vector of the Gaussian distribution.
    - `P` The covariance matrix of the Gaussian distribution.
"""
function from_moment(μ::Vector, P)
    @assert length(μ) == size(P, 1) "mean and covariance size mismatch"
    return Gaussian(μ, Matrix(P))
end 

""" 
    from_sqrt_moment(μ, S) 
    Construct a Gaussian distribution with given mean and square-root covariance matrix.

    # Arguments 
    - `μ` The mean vector of the Gaussian distribution.
    - `S` The square-root covariance matrix (upper triangular) of the Gaussian distribution.

"""
function from_sqrt_moment(μ::Vector, S)
    @assert length(μ) == size(S, 1) "mean and covariance size mismatch"
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
    @assert length(η) == size(Λ, 1) "mean and covariance size mismatch"
    return Gaussian(η, Matrix(Λ))
end 

""" 
    from_sqrt_info(ν, Ξ) 
    Construct a Gaussian distribution in square-root form with given information vector and information matrix.

    # Arguments 
    - `ν` The information vector of the Gaussian distribution.
    - `Ξ` The square-root information matrix (upper triangular) of the Gaussian distribution.

"""
function from_sqrt_info(ν::Vector, Ξ)
    @assert length(ν) == size(Ξ, 1) "mean and covariance size mismatch"
    @assert istriu(Matrix(Ξ)) "Ξ must be upper triangular"
    return Gaussian(ν, Matrix(Ξ))
end 


"""
    log_pdf(x::Vector, distribution::Gaussian; grad::Bool, sqrt::Bool) 

Compute the logarithm of a multivariate normal distribution at the value `x`. 

# Arguments
- `x` The input vector at which to evaluate the log-likelihood.
- `pdf` A multivariate normal distribution with mean `μ` and covariance matrix `Σ` or square-root covariance matrix `S` such that SᵀS = Σ.

# Keyword Arguments
- `grad`: If `true`, also compute the gradient of the log-likelihood with respect to `x`.
- `sqrt`: If `true`, the covariance matrix is given in square-root form `S` such that SᵀS = Σ.

# Returns
- The log of the probability distribution function evaluated at `x`.
"""
function log_pdf(x, distribution::Gaussian; grad=grad, sqrt=sqrt)
    
    μ = distribution.mean
    @assert length(x) == length(μ) "Input x and mean μ must have same length"
    n = length(x)

    Δ = x .- μ  

    if sqrt
        
        S = distribution.covariance
    
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
    conditional(distribution::Gaussian, idx_𝑥::Int, idx_𝑦::Int, 𝑦)

Given the joint Gaussian N(μ, Σ) and index sets for variables A and B, return 𝑝(𝑥 | 𝑦).
The joint distribution passed to this function must be in the form p([𝑦; 𝑥]) and not p([𝑥; 𝑦]) 

# Arguments
- `distribution::Gaussian`: The joint probability distribution function as a Gaussian distribution 𝑝(𝑥, 𝑦).
- `idx_𝑥::Vector`: The indices of the variables to condition on.
- `idx_𝑦::Vector`: The indices of the variables to condition on.
- `𝑦::Vector`: The values of the variables to condition on.

# Returns
- `μ_cond::Vector`: Conditional mean μ_A|B
- `Σ_cond::Matrix`: Conditional covariance Σ_A|B
"""
function conditional(distribution, idx_𝑥, idx_𝑦, 𝑦; sqrt=false)

    if sqrt

        # The conditional distribution of 𝑦 given 𝑥 is given by 𝑝(𝑦 | 𝑥) = 𝑁(μ𝑦 + S₂ᵀS₁⁻ᵀ(𝑥 - μ𝑥), S₃)
        μ = distribution.mean
        S = distribution.covariance 
        
        # Extract the blocks S₁, S₂, S₃ from S, this assumes that the square-root covariance is stored as S and not Sᵀ
        S₁ = S[idx_𝑦, idx_𝑦]
        S₂ = S[idx_𝑦, idx_𝑥]
        S₃ = S[idx_𝑥, idx_𝑥]

        # Compute S₁⁻ᵀ(𝑥 - μ𝑥) by solving the linear system S₁ * w = 𝑦 - μ𝑦
        w = S₁ \ (𝑦 - μ[idx_𝑦])
        
        # Compute the conditional mean μ_cond = μ𝑦 + S₂ᵀS₁⁻ᵀ(𝑦 - μ𝑦)
        μ_cond = μ[idx_𝑥] + S₂' * w

        # Compute the conditional square-root covariance S_cond = S₃, that is the square-root covariance of p(𝑦 | 𝑥)
        S_cond = S₃

        return from_sqrt_moment(μ_cond, S_cond)

    else

        μ = distribution.mean
        Σ = distribution.covariance
        
        μ𝑥 = μ[idx_𝑥]
        μ𝑦 = μ[idx_𝑦]

        Σ𝑥𝑥 = Σ[idx_𝑥, idx_𝑥]
        Σ𝑥𝑦 = Σ[idx_𝑥, idx_𝑦]
        Σ𝑦𝑥 = Σ[idx_𝑦, idx_𝑥]
        Σ𝑦𝑦 = Σ[idx_𝑦, idx_𝑦]

        # Compute the new mean and covariance of the conditional distribution 𝑝(𝑥 | 𝑦)
        # Dont invert the matrix (Σ𝑦𝑦⁻¹) - https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/

        # Instead, solve the linear system Σ𝑦𝑦 * w = 𝑦 - μ𝑦 to find w = Σ𝑦𝑦⁻¹ * (𝑦 - μ𝑦) 
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
    marginal(distribution::Gaussian, idx::Vector{Int})

Compute the marginal distribution of a Gaussian distribution over a subset of variables.

# Arguments
- `distribution::Gaussian`: The Gaussian distribution to marginalize.
- `idx::Vector{Int}`: The indices of the variables to marginalise over.
"""
function marginal(distribution::Gaussian, idx::Vector{Int}; sqrt=sqrt)
    if sqrt
        #            Σ = SᵀS

        # [ Σ𝑥𝑥  Σ𝑥𝑦 ] = [  S₁  S₂ ]ᵀ [ S₁  S₂ ]
        # [ Σ𝑦𝑥  Σ𝑦𝑦 ] = [  0   S₃ ]  [ 0   S₃ ]

        #              = [ S₁ᵀ  0  ]  [ S₁  S₂ ]
        #              = [ S₂ᵀ  S₃ᵀ]  [ 0   S₃ ]

        #              = [ S₁ᵀS₁      S₁ᵀS₂     ]
        #              = [ S₂ᵀS₁  S₂ᵀS₂ + S₃ᵀS₃ ]

        S𝑥𝑥 = distribution.covariance
        S₂ = S𝑥𝑥[idx, idx:end]
        S₃ = S𝑥𝑥[idx:end, idx:end]

        R₁ = qr([S₂; S₃]).R

        return from_sqrt_moment(distribution.mean[idx], R₁)

    else 
        return from_moment(distribution.mean[idx], distribution.covariance[idx, idx])
    end 
end


"""
join(distribution_𝑥, distribution_𝑦; sqrt=false)

Construct the joint distribution of two independent Gaussian densities.

# Arguments
- `distribution_𝑥`: A Gaussian distribution representing the first random variable.
- `distribution_𝑦`: A Gaussian distribution representing the second random variable.

# Keyword Arguments
- `sqrt`: If `true`, constructs the joint in square-root form (not yet implemented). Defaults to `false`.

# Returns
- A new Gaussian representing the joint distribution, with concatenated means and a block-diagonal covariance matrix.
"""
# TODO: Finish implementation 
function join(distribution_𝑥, distribution_𝑦; sqrt=false)

    μ = vcat(distribution_𝑥.mean, distribution_𝑦.mean)
   
    n𝑥 = size(distribution_𝑥.covariance, 1)
    n𝑦 = size(distribution_𝑦.covariance, 1)

    if sqrt
        error("Not implemented yet") # TODO
        return from_sqrt_moment(μ, S)
    else 
        # Create block diagonal matrix from the two covariance matrices
        Σ = zeros(n𝑥 + n𝑦, n𝑥 + n𝑦)
        Σ[1:n𝑥, 1:n𝑥] = distribution_𝑥.covariance
        Σ[n𝑥+1:end, n𝑥+1:end] = distribution_𝑦.covariance
        return from_moment(μ, Σ)
    end 
end


function add(𝑥₁::Gaussian, 𝑥₂::Gaussian; sqrt=sqrt)
    
    @assert length(𝑥₁.mean) == length(𝑥₂.mean) "mean length mismatch"
    @assert size(𝑥₁.covariance) == size(𝑥₂.covariance) "covariance size mismatch"

    μ = 𝑥₁.mean + 𝑥₂.mean

    if sqrt
        
        # Prepare the matrix for QR decomposition
        A = vcat(𝑥₁.covariance, 𝑥₂.covariance)

        # Perform QR decomposition and extract the upper triangular matrix R, by default the QR decomposition returns the upper square non-zero part of the matrix
        S = qr(A).R

        return from_sqrt_moment(μ, S)
    else 
        
        Σ = 𝑥₁.covariance + 𝑥₂.covariance
        
        return from_moment(μ, Σ)
    end 
end 


"""
    unscented_transform(func::Any, distribution::Gaussian; sqrt=sqrt)
    Perform the Unscented Transform (UT) of a Gaussian random variable through a nonlinear function `func`.

    # Arguments
    - `func`: The nonlinear function to be applied to the Gaussian random variable.
    - `distribution`: The Gaussian random variable to be transformed.
    
    # Keyword Arguments
    - `sqrt`: If `true`, the covariance matrix is given in square-root form `S` such that SᵀS = Σ.
"""
function unscented_transform(func::Any, distribution::Gaussian; sqrt=sqrt)
    if sqrt
        error("Not implemented yet") # TODO
    else
        μ𝑥 = distribution.mean
        Σ𝑥 = distribution.covariance
        L = length(μ𝑥)
        
        # UKF parameters
        κ = 0
        α = 1                  
        β = 2
        λ = α^2 * (L + κ) - L

        Sₓ = cholesky((L + λ) * Σ𝑥).L
        
        𝛘 = zeros(Float64, L, 2L + 1)
        𝛘[:, 1] = μ𝑥
        
        for i in 1:L
            𝛘[:, i+1] = μ𝑥 + Sₓ[:, i]
            𝛘[:, i+1+L] = μ𝑥 - Sₓ[:, i]
        end

        # Weights 
        𝑾ᵐ = zeros(2L + 1)
        𝑾ᶜ = zeros(2L + 1)
        𝑾ᵐ[1] = λ / (L + λ)
        𝑾ᶜ[1] = λ / (L + λ) + (1 - α^2 + β)
        𝑾ᵐ[2:end] .= 1 / (2 * (L + λ))
        𝑾ᶜ[2:end] .= 1 / (2 * (L + λ))

        # Transform sigma points through measurement model
        μ𝑦 = func(𝛘[:, 1])                      
        n𝑦 = length(μ𝑦)
        𝒴 = zeros(n𝑦, 2L + 1)    

        𝒴[:, 1] = μ𝑦
        
        for i in 2:(2L + 1)    
            𝒴[:, i] = func(𝛘[:, i])
        end

        # μ𝑦 = sum(𝑾ᵐ[i] * 𝒴[i] for i in 1:(2L + 1))
        # Σ𝑦 = sum(𝑾ᶜ[i] * (𝒴[i] - μ𝑦)^2 for i in 1:(2L + 1))
        
        # # Compute cross-covariance (state-measurement)
        # Σ𝑥𝑦 = sum(𝑾ᶜ[i] * (𝛘[:, i] - μ𝑥) * (𝒴[i] - μ𝑦) for i in 1:(2L + 1))

        # Clean and efficient way to compute the mean and covariance 
        # TODO: Review the code below and prove why it is equivalent to the code above

        μ𝑦 = 𝒴 * 𝑾ᵐ
        dY = 𝒴 .- μ𝑦
        Σ𝑦 = dY * Diagonal(𝑾ᶜ) * dY'
        Σ𝑦 = 0.5 * (Σ𝑦 + Σ𝑦')

        return from_moment(μ𝑦, Σ𝑦)
    end
end 


"""
    This method transforms the Gaussian distribution p(𝑥) through a nonlinear function y = f(𝑥) by 
    propogating information through the affine transformation. It returns a new Gaussian distribution
    representing p(𝑦)

    # Arguments
    - `func`: 
    - `distribution`: The Gaussian distribution to be propogated through the nonlinear function.

    # Returns
    - `p(𝑦)`: The transformed Gaussian distribution.

"""
function affine_transform(func::Any, distribution::Gaussian; sqrt=sqrt)

    # The notation 𝑦 represent the output distribution of the affine transformation and should not be confused with the distribution the measurement likelihood 𝑝(𝑦) 
    μ𝑥 = distribution.mean

    # Evalute h(μx) to obtain μy 
    μ𝑦 = func(μ𝑥)

    # Evalute ∂h(x)/∂x at x = μ, that is the Jacobian of h evalutated at μ
    C = ForwardDiff.jacobian(func, μ𝑥)
    
    if sqrt

        # S𝑦𝑦ᵀS𝑦𝑦 = C * S𝑥𝑥ᵀS𝑥𝑥 * C' = (S𝑥𝑥Cᵀ)ᵀ(S𝑥𝑥Cᵀ)
        S𝑥𝑥 = distribution.covariance

        # If the output dimension is the same as the state dimension then we know its a prediction step 
        if length(μ𝑦) == 3 
            S𝑦𝑦 = qr(S𝑥𝑥*C').R
        else
            # S𝑦𝑦ᵀS𝑦𝑦 = C * S𝑥𝑥ᵀS𝑥𝑥 * C' + SRᵀSR = (S𝑥𝑥Cᵀ)ᵀ(S𝑥𝑥Cᵀ) + SRᵀSR 
            SR = Matrix(zeros(4, 4))
            SR[1, 1] = 50.0

            # Ensure S𝑦𝑦 is upper triangular via QR decomposition
            S𝑦𝑦 = qr([(S𝑥𝑥*C'); SR]).R 
        end
        return from_sqrt_moment(μ𝑦, S𝑦𝑦) 
    else
        
        Σ𝑥𝑥 = distribution.covariance 
        Σ𝑦𝑦 = C * Σ𝑥𝑥 * C'
        
        @assert isapprox(Σ𝑦𝑦, Σ𝑦𝑦', rtol=1e-6) "Covariance not symmetric"
    
        return from_moment(μ𝑦, Σ𝑦𝑦)
    end 
end    
