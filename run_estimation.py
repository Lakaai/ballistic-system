import numpy as np
import pandas as pd
from src.gaussian import Gaussian

df = pd.read_csv("ballistic_data.csv")
t_hist = df[:, 0]
x_hist = df[:, 1:3]
y_hist = df[:, 5]
mu0 = np.array[]
sigma0 = np.array[]

initial_density = Gaussian.from_moment(mu0, sigma0)

def run_estimation(measurements, update_method):

