import numpy as np


def r2_score(y, x):
    A = np.vstack([x, np.ones(len(x))]).T

    # Use numpy's least squares function
    m, c = np.linalg.lstsq(A, y)[0]

    # print(m, c)
    # 1.97 -0.11

    # Define the values of our least squares fit
    f = m * x + c

    # print(f)
    # [ 1.86  3.83  5.8   7.77  9.74]

    # Calculate R^2 explicitly
    yminusf2 = (y - f)**2
    sserr = sum(yminusf2)
    mean = float(sum(y)) / float(len(y))
    yminusmean2 = (y - mean)**2
    sstot = sum(yminusmean2)
    R2 = 1. -(sserr / sstot)
    return R2

# x = np.arange(1, 6, 1)
# y = np.array([1.9, 3.7, 5.8, 8.0, 9.6])
# print(x, r2score(x, y))
