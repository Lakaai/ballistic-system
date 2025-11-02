using CSV
using DataFrames
using LinearAlgebra
using Infiltrator
using Plots

include("system_estimator.jl")
include("gaussian.jl")

data = CSV.read("data/estimationdata.csv", DataFrame)

nx = 3  # Dimension of state vector
ny = 1  # Dimension of measurement vector

t_hist = data[:, 1]         # Time stamps
x_hist = data[:, 2:4]       # State data
y_hist = data[:, 6]         # Measurement data 

mu0 = [14000.0; -450.0; 0.0005]                     # Initial state estimate
S0 = Matrix(Diagonal([2200.0, 100.0, 1e-3]))        # Initial covariance estimate

function run_sqrt_filter(t_hist, y_hist, mu0, S0)

    density = from_sqrt_moment(mu0, S0)

    nsteps = nrow(data)     # Number of time steps
    μ_hist = [] 
    S_hist = []

    for i = 1:nsteps

        time = t_hist[i]                # Get the current time
        measurement = [y_hist[i]]       # Get the current measurement and convert to vector type since estimation machinery expects vector valued measurement
        
        println("Time: ", time)

        # Form the predicted density p(𝑥ₖ ∣ 𝑦ₖ₋₁) by propagating p(𝑥ₖ₋₁ ∣ 𝑦ₖ₋₁) through the process model 
        density = predict(time, density, AFFINE; sqrt=true)         

        # Compute the filtered density p(𝑥ₖ ∣ 𝑦₁:𝑦ₖ)
        density = update(density, measurement, AFFINE; sqrt=true)   
        
        # Store the data for plotting, taking the absolute value of covariance since square-root factorisation can be negative
        push!(μ_hist, density.mean)
        push!(S_hist, sqrt.(diag(abs.(density.covariance))))        

    end 
    return μ_hist, S_hist
end 

μ_hist, S_hist = run_sqrt_filter(t_hist, y_hist, mu0, S0) 

# Convert to matrices for easier plot handling
mu_matrix = hcat(μ_hist...)
covariance_matrix = hcat(S_hist...)

gr()   # Plotting backend

# Left column: state estimates
p1 = plot(t_hist, mu_matrix[1, :], label="μ₁", ylabel="Altitude [m]", title="State Estimates", lw=2, legend=:topright)
plot!(p1, t_hist, x_hist[:, 1], label="x₁ (true)", color=:red, lw=2, linestyle=:dash)

p2 = plot(t_hist, mu_matrix[2, :], label="μ₂", ylabel="Velocity [m/s]", lw=2, legend=:topright)
plot!(p2, t_hist, x_hist[:, 2], label="x₂ (true)", color=:red, lw=2, linestyle=:dash)

p3 = plot(t_hist, mu_matrix[3, :], label="μ₃", ylabel="Drag Coeff", lw=2, legend=:topright)
plot!(p3, t_hist, x_hist[:, 3], label="x₃ (true)", color=:red, lw=2, linestyle=:dash)

# Right column: marginal standard deviations σᵢ = √Σᵢᵢ
p4 = plot(t_hist, covariance_matrix[1, :], label="S₁", title="Marginal Standard Deviations", ylabel="Altitude Std [m]", lw=2)
p5 = plot(t_hist, covariance_matrix[2, :], label="S₂", ylabel="Velocity Std [m/s]", lw=2)
p6 = plot(t_hist, covariance_matrix[3, :], label="S₃", ylabel="Drag Coeff Std", lw=2)

# Add x-axis label only to bottom plots
xlabel!(p3, "Time [s]")
xlabel!(p6, "Time [s]")

plot(p1, p4, p2, p5, p3, p6, layout=(3, 2), size=(800, 800))



