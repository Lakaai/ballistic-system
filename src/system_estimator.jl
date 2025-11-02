using LinearAlgebra
using ForwardDiff
using Optim
using Infiltrator

include("measurement.jl")

@enum UpdateMethod AFFINE UNSCENTED NEWTONTRUSTEIG BFGSTRUST

const p0 = 101.325e3       # Air pressure at sea level [Pa]
const M  = 0.0289644       # Molar mass of dry air [kg/mol]
const R  = 8.31447         # Gas constant [J/(mol·K)]
const L  = 0.0065          # Temperature gradient [K/m]
const T0 = 288.15          # Temperature at sea level [K]
const g  = 9.81            # Acceleration due to gravity [m/s²]
time_ = 0                  # Previous time stamp 





function predict(time::Any, density::Gaussian, update_method::UpdateMethod; sqrt=sqrt)
    global time_
    dt = time - time_   # Update the time increment
    time_= time         # Update the previous time

    @assert dt >= 0 "dt must be non-negative"

    if dt == 0  
        return density
    end 

    # Define process noise covariance Q
    Q = Matrix(Diagonal([1e-20, 25e-12, 0.0]))  # [velocity, drag coeff, altitude (no noise)]

    process_model = μ𝑥 -> rk4_step(μ𝑥, dt)
    
    if update_method == UNSCENTED

        predicted_density = unscented_transform(process_model, density; sqrt=sqrt)
        predicted_density = from_moment(predicted_density.mean, predicted_density.covariance + Q)   # Add process noise after transformation

    elseif update_method == AFFINE

        predicted_density = affine_transform(process_model, density; sqrt=sqrt)
        predicted_density = from_moment(predicted_density.mean, predicted_density.covariance + Q)   # Add process noise after transformation

    else
        error("Invalid prediction method: $update_method")
    end  

    return predicted_density

end 


"""
    rk4_step(x::Vector{Float64}, dt::Float64) -> Vector{Float64}

Propagates the system state `xₖ` forward one time step using the classical Runge-Kutta 4 (RK4) integration method.
Map x[k] to x[k+1] using RK4 integration
This version assumes a **deterministic process model** (`dx = f(x) dt`) with no process noise.

# Arguments
- `xₖ`: Current state vector.
- `dt`: Time step duration.

# Returns
- `xₖ₊₁`: Estimated state at the next time step.
"""
function rk4_step(xₖ::Any, dt::Any)
    k1 = dynamics(xₖ)
    k2 = dynamics(xₖ .+ 0.5 .* dt .* k1)
    k3 = dynamics(xₖ .+ 0.5 .* dt .* k2)
    k4 = dynamics(xₖ .+ dt .* k3)

    xₖ₊₁ = xₖ .+ dt/6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)

    return xₖ₊₁ 
end 
 

"""
    rk4_sde_step(xdw::Vector{Float64}, dt::Float64, idxQ::Vector{Int}, augmented_dynamics::Function) -> (Vector{Float64}, Matrix{Float64})

Propagates the augmented state `[xₖ; Δwₖ]` forward one time step using an RK4 method for stochastic systems with additive noise.

This mirrors a typical **SDE-based system update** used in filters that compute Jacobians with respect to both the state and process noise. The function supports:
- RK4 integration of the state and its derivatives.
- Process noise influence using indices in `idxQ`.
- Jacobian computation for use in filters like the EKF.

# Arguments
- `xdw`: Augmented input vector `[x; dw_subset]`, where `x` is the state and `dw_subset` is a subset of the process noise.
- `dt`: Fixed time step.
- `idxQ`: Indices specifying which noise components affect which states.
- `augmented_dynamics`: Function computing the augmented system dynamics `f(X)`, where `X` contains the state and partial derivatives.

# Returns
- `Xₖ₊₁` The object that contains [xₖ₊₁ ∂xₖ₊₁/∂xₖ  ∂xₖ₊₁/∂Δwₖ].
"""
function rk4_sde_step(xₖ::Any, Δt::Any)

    nx = length(xdw) - length(idxQ)
    nq = length(idxQ)

    x = xdw[1:nx]
    dw = zeros(nx)
    dw[idxQ] .= xdw[nx+1:end]

    # Augmented matrices: [xₖ₊₁ ∂xₖ₊₁/∂xₖ  ∂xₖ₊₁/∂Δwₖ]
    Xₖ  = hcat(x, I(nx), zeros(nx, nx))
    ΔWₖ = hcat(dw, zeros(nx, nx), I(nx))

    F1 = augmented_dynamics(Xₖ)
    F2 = augmented_dynamics(Xₖ .+ (F1 .* Δt .+ ΔWₖ) ./ 2)
    F3 = augmented_dynamics(Xₖ .+ (F2 .* Δt .+ ΔWₖ) ./ 2)
    F4 = augmented_dynamics(Xₖ .+ F3 .* Δt .+ ΔWₖ)

    Xₖ₊₁ = Xₖ .+ (F1 .+ 2 .* F2 .+ 2 .* F3 .+ F4) .* (Δt / 6) .+ Δwₖ

    # Extract Jacobian
    J = hcat(Xₖ₊₁[:, 2:nx+1], Xₖ₊₁[:, nx+1 .+ idxQ])

    return Xₖ₊₁[:, 1], J
end 
    

# Evaluate f(x) from the SDE dx = f(x)*dt + dw
function dynamics(x::Any; jacobian=false, hessian=false)

    # Extract state variables
    h = x[1]
    v = x[2]
    c = x[3]
    
    f = similar(x)

    # Calculate temperature at altitude h
    T = T0 - L * h
    
    d = ((0.5 * M * p0) / R) * (1 / T) * (1 - L * h / T0)^(g * M / (R * L)) * v^2 * c

    # Set f according to the dynamics equations
    f[1] = v;                 # dh/dt = v
    f[2] = d - g;             # dv/dt = d - g
    f[3] = 0;                 # dc/dt = 0 (drag coefficient is constant)

    # If the neither the jacobian or hessian is required then only return f
    if !jacobian && !hessian
        return f
    end 

    if jacobian
        # Calculate partial derivatives
        dd_dh = ((0.5 * M * p0) / R) * v^2 * c * (
            (L / (T^2)) * (1 - L * h / T0)^(g * M / (R * L)) -
            (g * M / (R * T0)) * (1 - L * h / T0)^((g * M / (R * L)) - 1) / T)
        dd_dv = 2 * d / v
        dd_dc = d / c

        # Resize J to the correct size and fill in values
        J = zeros(length(f), length(x))

        J[1, 2] = 1.0
        J[2, 1] = dd_dh
        J[2, 2] = dd_dv
        J[2, 3] = dd_dc
        # J[3, :] stays zero
        
        if !hessian
            return f, J
        end 

    if hessian
        # TODO Implement hessian
        error("Hessian not yet implemented")
    end 

end 

    return f
end

# Evaluate F(X) from dX = F(X)*dt + dW
function augmentedDynamics(X::Any)
    @assert size(X, 1) > 0 "X must have at least one row"
    nx = size(X, 1)

    x = X[:, 1]
    f, J = dynamics(x, jacobian=true)

    @assert size(f, 1) == nx "f must have nx rows"
    @assert size(J, 1) == nx && size(J, 2) == nx "J must be nx by nx"

    dX = hcat(f, J * X[:, 2:end])
    return dX
end 


"""
    cost_function_factory(density::Gaussian, measurement)

The mean and covariance values must be available at the time of evaluation of the cost function. However, the cost function 
passed to the optimiser requires a function signature of f(x), therefore we will create a function factory to create a cost function 
with the required signature whilst still having access to the mean and covaraince values. 

In general a `factory` is an object for creating other objects, in this case it is a function that returns another function.

# Arguments
- `density` The system density.
- `measurement` The current measurement vector.

# Returns
- A function with signature f(x) that can be passed to the optimiser.

"""
function cost_function_factory(density::Gaussian, measurement; sqrt=sqrt)
    return function(x) # Returns a cost function f(x) which has the required signature for the optimiser
        logprior = log_pdf(x, density; grad=false, sqrt=sqrt)

        # You must define this based on your measurement model
        loglik, _ = logLikelihood(x, measurement; grad=true, sqrt=sqrt)

         # Return a scalar cost (negative log-likelihood (the measurement cost) + log-prior (the prediction cost))
        return -(logprior + loglik) # Return −logp(x∣z) = -(logp(x) + logp(z∣x))
    end
end


function measurement_update_bfgs(density::Gaussian, measurement::Any; sqrt=sqrt)
    if sqrt
        error("Not implemented yet") # TODO: Implement square root BFGS update
        x0 = density.mean
        S = density.covariance

        # Laplace Aproximation ℐ = ∫ₓf(x)dx ≈ f(x*) √det(2*π*P)
        
        df = TwiceDifferentiable(cost_function_factory(density, measurement; sqrt=sqrt), x0, autodiff = :forward) # Store and reuse gradient and hessian 
        res = optimize(df, x0, BFGS())
        @assert res.converged "Optimiser has not converged."

        # x* = argmax f(x)
        x_map = Optim.minimizer(res)

        # Posterior sqrt covariance approximation (naive)
        H = ForwardDiff.hessian(cost_function_factory(density, measurement; sqrt=sqrt), x_map)

        F = cholesky(H)   # H = F'U F, F.U is upper triangular
        S = Matrix(inv(F.U))      # S * S' = H^{-1}
        
        return Gaussian(x_map, S)
    else 
        x0 = density.mean
        Σ = density.covariance

        df = TwiceDifferentiable(cost_function_factory(density, measurement; sqrt=sqrt), x0, autodiff = :forward) # Store and reuse gradient and hessian 
        res = optimize(df, x0, BFGS())

        x_map = Optim.minimizer(res)

        # Posterior sqrt covariance approximation (naive)
        H = ForwardDiff.hessian(cost_function_factory(density, measurement; sqrt=sqrt), x_map)
         
        Σ = Matrix(inv(H))     
        
        return Gaussian(x_map, Σ)
    end 
end 

function measurement_update_unscented(density::Gaussian, measurement::Any; sqrt=sqrt)

    if sqrt
        # Form the joint probability density 𝑝(𝑥ₖ, 𝑦ₖ | 𝑦₁...𝑦ₖ₋₁), that is the probability of the state 𝑥ₖ and the measurement 𝑦ₖ given all past measurements 𝑦₁, 𝑦₂, ..., 𝑦ₖ₋₁
        transformed_density = unscented_transform(predict_measurement, density; sqrt=sqrt)

        # Condition on the measurement 𝑦ₖ to form the posterior density 𝑝(𝑥ₖ | 𝑦ₖ)
        return conditional(transformed_density, 2:4, 1:1, measurement; sqrt=sqrt)
        
    else 

        # Measurement noise covariance 
        # noise_density = from_moment(0, Matrix(Diagonal([50.0^2])))

        # density = join(density, noise_density)

        # Form the joint probability density 𝑝(𝑥ₖ, 𝑦ₖ | 𝑦₁...𝑦ₖ₋₁), that is the probability of the state 𝑥ₖ and the measurement 𝑦ₖ given all past measurements 𝑦₁, 𝑦₂, ..., 𝑦ₖ₋₁
        transformed_density = unscented_transform(augmented_predict_measurement, density; sqrt=sqrt)

        μ = transformed_density.mean
        Σ = transformed_density.covariance
        
        R = Matrix(Diagonal([50.0^2]))                    
        Σ[1:1, 1:1] += R                        # Measurement block
        Σ = 0.5 * (Σ + Σ')                      # Symmetrise
        
        transformed_density = from_moment(μ, Σ)

        # Condition on the measurement 𝑦ₖ to form the posterior density 𝑝(𝑥ₖ | 𝑦ₖ)
        return conditional(transformed_density, 2:4, 1:1, measurement; sqrt=sqrt)
    end 
end 

function measurement_update_affine(density::Gaussian, measurement::Any; sqrt=sqrt)
    if sqrt
        display(density.covariance)
        transformed_density = affine_transform(augmented_predict_measurement, density; sqrt=sqrt)

        μ = transformed_density.mean
        S = transformed_density.covariance
        
        transformed_density = from_sqrt_moment(μ, S)

        # Condition on the measurement 𝑦ₖ to form the posterior density 𝑝(𝑥ₖ | 𝑦ₖ)
        return conditional(transformed_density, 2:4, 1:1, measurement; sqrt=sqrt)
    else 
        
        transformed_density = affine_transform(augmented_predict_measurement, density; sqrt=sqrt)

        μ = transformed_density.mean
        Σ = transformed_density.covariance
        
        R = Matrix(Diagonal([50.0^2]))                    
        Σ[1:1, 1:1] += R                        # Measurement block
        Σ = 0.5 * (Σ + Σ')                      # Symmetrise
        
        transformed_density = from_moment(μ, Σ)

        # Condition on the measurement 𝑦ₖ to form the posterior density 𝑝(𝑥ₖ | 𝑦ₖ)
        return conditional(transformed_density, 2:4, 1:1, measurement; sqrt=sqrt)
    end 
end 

function update(density::Gaussian, measurement::Any, update_method::UpdateMethod; sqrt=sqrt)
    if update_method == BFGSTRUST
        density = measurement_update_bfgs(density, measurement; sqrt=sqrt)
    elseif update_method == UNSCENTED
        density = measurement_update_unscented(density, measurement; sqrt=sqrt) 
    elseif update_method == AFFINE
        density = measurement_update_affine(density, measurement; sqrt=sqrt)
    elseif update_method == NEWTONTRUSTEIG
        error("Not implemented yet")
        # TODO: density = measurement_update_newtontrusteig(density, measurement)
    else
        error("Invalid update method: $update_method")
    end  
end  