import numpy as np
import matplotlib.pyplot as plt

data_list = []
for iter in range(100):
    random_idx = np.random.randint(0, 2)
    if random_idx == 0:
        data = np.random.normal((0.5,0), 0.1, 2)
    elif random_idx == 1:
        data = np.random.normal((-0.5,0), 0.1, 2)
    data_list.append(data)

data_list = np.array(data_list)
plt.scatter(data_list[:, 0], data_list[:, 1])
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.show()


data_list = []
for iter in range(100):
    x = np.random.uniform(-1, 1)
    y = x + np.random.normal(0, 0.1)
    data = [x, y]
    data = np.array(data)
    data_list.append(data)

data_list = np.array(data_list)
plt.scatter(data_list[:, 0], data_list[:, 1])
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.show()