import numpy as np

OCT_DEFAULT_MEAN = (0.3124, 0.3124, 0.3124)
OCT_DEFAULT_STD = (0.2206, 0.2206, 0.2206)

from scipy.stats import wilcoxon
# i1 = [1, 0, 0, 1, 0, 0, 0, 1, 1, 1]
# i2 = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
data = np.array([97.55, 81.95, 91.74, 96.16, 98.53]) - 91.85
print(data)
w, p = wilcoxon(data, correction=True)
print(p)