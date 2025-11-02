# Ballistic System 

## Overview 

This repository contains an example of non-linear state estimation applied to a simple ballistic system. By setting the update method keyword in calls to 'predict' and 'update' you can switch between Affine and Unscented transformations leading to the Extended Kalman filter and Unscented Kalman filters respectively. 
 

The implementation includes gaussian.jl which provides several functions for common Gaussian operations including conditioning, marginalisation, forming joint probability denisty functions and computing the logarithm of a probability density function.  

## Setup

```bash
git clone <this-repo>
cd BallisticSystem
julia --project -e 'using Pkg; Pkg.instantiate()'
```

This will install all dependencies listed in `Project.toml`.

## WIP Features
Square root covariance implementation.
