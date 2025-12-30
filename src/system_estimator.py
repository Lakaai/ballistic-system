from src.gaussian import Gaussian
from src.system_wamv import dynamics

previous_time = 0

def dynamics():
    return 0 

def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + k1 * dt / 2)
    k3 = f(x + k2 * dt / 2)
    k4 = f(x + k3 * dt)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

def predict(current_time, density, update_method):

    global previous_time
    dt = current_time - previous_time
    previous_time = current_time 

    if dt == 0:
        return density
    
    process_model = rk4_step(dynamics(), density, dt)

    if update_method == "UNSCENTED":
        return Gaussian.unscented_transform(process_model, density)
    else:
        raise ValueError(f"Invalid update method: {update_method}")
    pass

def update(density, update_method):

    if update_method == "UNSCENTED":

        # Form the joint density p(xk, yk | y1:k-1) by propagating the prior p(xk | y1:k-1) through the transformation
        pxy = Gaussian.unscented_transform(density)

        # Compute the conditional density p(xk | yk) by conditioning the joint density on the measurement yk
        return Gaussian.conditional(pxy)

    elif update_method == "AFFINE":
        raise NotImplementedError("Affine update method not implemented")

    else:
        raise ValueError(f"Invalid update method: {update_method}")
    