import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

input = []
case=3
for iter in range(20):
    if case == 0:
        random_idx = np.random.randint(0, 2)
        if random_idx == 0:
            data = np.random.normal((-0.5, 0), 0.1, 2)
        elif random_idx == 1:
            data = np.random.normal((0.5, 0), 0.1, 2)
    elif case == 1:
        random_idx = np.random.randint(0, 4)
        data = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        data = np.array(data)
        data = data[random_idx]
    elif case == 2:
        x = np.random.uniform(-0.3, 0.3)
        y = x + np.random.normal(0, 0.05)
        data = [x, y]
        data = np.array(data)
    elif case ==3:
        y = np.random.uniform(-0.8, 0.8)
        x = np.random.normal(0, 0.05)
        data = [x, y]
        data = np.array(data)

    input.append(data)


# pca input
#plot data
input = np.array(input)
plt.scatter(input[:, 0], input[:, 1])
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.show()

pca = PCA(n_components=1)
pca.fit_transform(input)
print (pca.explained_variance_ratio_)
print (pca.components_)



# input = np.array(input)
# input = input.T
# input = input - np.mean(input, axis=1, keepdims=True)
# cov = np.dot(input, input.T)
# eigenvalue, eigenvector = np.linalg.eig(cov)
# idx = eigenvalue.argsort()[::-1]
# eigenvalue = eigenvalue[idx]
# eigenvector = eigenvector[:, idx]
# eigenvector = eigenvector.T
# input = np.dot(eigenvector, input)
# input = input.T
# input = input / np.max(np.abs(input), axis=0, keepdims=True)
# input = input.T
