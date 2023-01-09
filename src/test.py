import numpy as np


a = np.ones((3, 1))
b = np.zeros((3,1))
c = np.ones((3, 1))

final = np.vstack((a, b, c))

print(final)