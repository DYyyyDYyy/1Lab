import numpy as np


a = 0.0
b = 2.0
y0 = 1.0

def f(x, y):
    return x - y

def y_exact(x):
    return x - 1 + 2 * np.exp(-x)