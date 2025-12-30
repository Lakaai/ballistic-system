import numpy as np
from gaussian import Gaussian

def test_conditional():

    mu = np.array([1.0, 2.0, 3.0, 4.0])  
    sigma = np.array([
    [1.0, 0.2, 0.1, 0.0],
    [0.2, 1.5, 0.3, 0.2],
    [0.1, 0.3, 2.0, 0.5],
    [0.0, 0.2, 0.5, 1.2]
    ])       

    idx_x = np.array([1, 2, 3])     
    idx_y = np.array([0])           
    y = np.array([0.8])     

    distribution = Gaussian.from_moment(mu, sigma)

    out = Gaussian.conditional(distribution, idx_x, idx_y, y, sqrt=False)

    expected_mean = np.array([1.96, 2.98, 4.0])
    expected_covariance = np.array([[
        1.46, 0.27999999999999997, 0.2],
        [0.27999999999999997, 1.99, 0.5],
        [0.2, 0.5, 1.2]
        ])

    assert np.allclose(out.mean, expected_mean)
    assert np.allclose(out.covariance, expected_covariance)

def test_unscented_transform():

    mu = np.array([1.0, 2.0, 3.0, 4.0])  
    sigma = np.array([
    [1.0, 0.2, 0.1, 0.0],
    [0.2, 1.5, 0.3, 0.2],
    [0.1, 0.3, 2.0, 0.5],
    [0.0, 0.2, 0.5, 1.2]
    ])       
         
    distribution = Gaussian.from_moment(mu, sigma)
    def nonlinear_func(x):
        return x**2

    out = Gaussian.unscented_transform(nonlinear_func, distribution, sqrt=False)

    expected_mean = np.array([2.0, 5.5, 11.0, 17.2])
    expected_covariance = np.array([
    [9.0, 3.2600000000000002, 3.240000000000001, 1.199999999999999],
    [3.2600000000000002, 34.782799999999995, 10.5152, 8.359999999999994],
    [3.240000000000001, 10.5152, 91.00898615124791, 27.258344905235496],
    [1.199999999999999, 8.359999999999994, 27.258344905235496, 82.80742110927032]
    ])

    assert np.allclose(out.mean, expected_mean)
    assert np.allclose(out.covariance, expected_covariance)


# def test_conditional_sqrt():

#     # Joint distribution p([y; x]) where y and x are 1D
#     mu = np.array([0.0, 0.0])  # Mean [mu_y, mu_x]
#     sigma = np.array([[2.0, 0.5], [0.5, 1.0]])  # Covariance [[var_y, cov_yx], [cov_xy, var_x]]
#     g = Gaussian(mu, sigma)

#     # Condition on x = 1.0
#     idx_x = [1]  # Index of x in the joint vector
#     idx_y = [0]  # Index of y in the joint vector
#     x_value = np.array([1.0])

#     cond = Gaussian.conditional(g, idx_x, idx_y, x_value, sqrt=False)

#     # Manually compute expected conditional mean and covariance
#     mu_y = mu[0]
#     mu_x = mu[1]
#     sigma_yy = sigma[0, 0]
#     sigma_yx = sigma[0, 1]
#     sigma_xy = sigma[1, 0]
#     sigma_xx = sigma[1, 1]

#     # Conditional mean: μ𝑦|𝑥 = μ𝑦 + Σ𝑦𝑥 * Σ𝑥𝑥⁻¹ * (𝑥 - μ𝑥)
#     expected_mu_cond = mu_y + sigma_yx * (1.0 - mu_x) / sigma_xx

#     # Conditional covariance: Σ𝑦|𝑥 = Σ𝑦𝑦 - Σ𝑦𝑥 * Σ𝑥𝑥⁻¹ * Σ𝑥𝑦
#     expected_sigma_cond = sigma_yy - sigma_yx * (1.0 / sigma_xx) * sigma_xy

#     assert np.allclose(cond.mean, expected_mu_cond)
#     assert np.allclose(cond.covariance, expected_sigma_cond)

# def test_constructor_and_from_moment():
#     mu = np.array([1.0, 2.0])
#     P = np.array([[1.0, 0.2], [0.2, 2.0]])

#     g1 = Gaussian(mu, P)
#     g2 = Gaussian.from_moment(mu, P)

#     assert np.allclose(g1.mean, mu)
#     assert np.allclose(g1.covariance, P)
#     assert np.allclose(g2.mean, mu)
#     assert np.allclose(g2.covariance, P)


# def test_marginal_returns_same():
#     mu = np.array([0.0, 0.0])
#     P = np.eye(2)
#     g = Gaussian(mu, P)

#     m = Gaussian.marginal(g)
#     assert np.allclose(m.mean, mu)
#     assert np.allclose(m.covariance, P)


# def test_unscented_transform_linear():
#     # For a linear function f(x) = A x + b, the UT should reproduce exact mean and covariance
#     A = np.array([[2.0, 0.0], [0.0, 0.5]])
#     b = np.array([0.1, -0.2])

#     def f(x):
#         return A @ x + b

#     mu = np.array([0.5, -1.0])
#     P = np.array([[0.3, 0.0], [0.0, 0.8]])
#     g = Gaussian(mu, P)

#     out = Gaussian.unscented_transform(f, g)

#     expected_mu = A @ mu + b
#     expected_P = A @ P @ A.T

#     assert np.allclose(out.mean, expected_mu, atol=1e-6)
#     assert np.allclose(out.covariance, expected_P, atol=1e-6)

