using Test
using Distributions
using LinearAlgebra

include("../src/system_estimator.jl")

@testset "system_estimator.jl" begin
            @testset "affine_transform" begin

                μ = [14000.0; -450.0; 0.0005]
                S𝑥𝑥 = Matrix(Diagonal([2200.0, 100.0, 1e-3]))   
                Σ𝑥𝑥 = Matrix(Diagonal([2200.0, 100.0, 1e-3].^2))
                
                process_model = μ -> rk4_step(μ, 0.1)
                
                # Obtain the transformed density in full covariance form 
                𝑦₁ = affine_transform(process_model, from_moment(μ, Σ𝑥𝑥); sqrt=false)

                # Obtain the transformed density in square root covariance form 
                𝑦₂ = affine_transform(process_model, from_sqrt_moment(μ, S𝑥𝑥); sqrt=true)
                
                # Form the full covariance matrix and compare to the density 𝑦₁ that was transformed in full covariance form 
                Σ𝑦𝑦 = 𝑦₂.covariance'*𝑦₂.covariance 
                
                @test isapprox(Σ𝑦𝑦, 𝑦₁.covariance; atol=1e-8)

                @test isapprox(𝑦₁.mean, [13955.012676670513, -449.7454001890395, 0.0005]; atol=1e-8)
                @test isapprox(𝑦₁.covariance, [4.840015788800788e6 155.558599234553 0.0001232275228907645; 155.55859923455276 9896.7853045739 0.0024644170003974594; 0.0001232275228907645 0.0024644170003974594 1.0e-6]; atol=1e-8)

            end
            @testset "unscented_transform" begin
               @test skip=true
               @test skip=true
            end
        end;