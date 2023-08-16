import numpy as np
import matplotlib.pyplot as plt


class Encoder:
    def __init__(self, input_dim, hidden_dim, lr):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Initialize weights and biases
        self.weights = np.random.uniform(-1, 1, input_dim)

    def forward(self, x):
        self.y = np.dot(x, self.weights)

    def backward(self, x):
        y2 = np.dot(self.y, self.y)
        delta_w = self.lr * ((np.dot(x, self.y) - np.dot(y2, self.weights)))
        self.weights += delta_w
        return self.weights


def loss_fxn(X, X_hat):
    return np.sum(np.abs(0.5 * ((X - X_hat) ** 2)))


def create_train_set(input_dim, size_train_set):
    train_set = []
    for iter in range(size_train_set):
        random_idx = np.random.randint(0, 3)
        if random_idx == 0:
            X = np.random.normal((-10, 10, -10), 1, input_dim)
        if random_idx == 1:
            X = np.random.normal((-10, -10, -10), 1, input_dim)
        if random_idx == 2:
            X = np.random.normal((20, 0, 20), 1, input_dim)
        train_set.append(X)
    return train_set


def train(encoder, train_set, eigenvector):
    for x in train_set:
        encoder.forward(x)
        encoder.backward(x)
        X_hat = np.dot(encoder.y, encoder.weights)
        loss = loss_fxn(x, X_hat)
        loss_list.append(loss)

        angle_to_eigenvector = np.degrees(np.arccos(
            np.dot(encoder.weights, eigenvector) / (np.linalg.norm(encoder.weights) * np.linalg.norm(eigenvector))))
        angle_list.append(angle_to_eigenvector)

        # Plot the input point
        ax.scatter(x[0], x[1], x[2], color='b', label='Input')

    ax.quiver(0, 0, 0, encoder.weights[0] * 10, encoder.weights[1] * 10, encoder.weights[2] * 10, color='r',
              label='Weights')
    ax.text(10, 15, 0, f'Weight: {np.round(encoder.weights, decimals=3)}', transform=ax.transAxes)
    ax.text(10, 15, -20, f'Eigenvector: {np.round(eigenvector, decimals=3)}', transform=ax.transAxes)


if __name__ == "__main__":
    INPUT_DIM = 3
    HIDDEN_DIM = 1
    LR = 0.001
    TRAIN_SET_SIZE = 100
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LR)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=11, azim=-154)

    z_lim = 20
    ax.set_xlim(-20, 25)
    ax.set_ylim(-20, 25)
    ax.set_zlim(-20, 25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['Weight', 'Input', 'Output'])

    loss_list = []
    angle_list = []
    train_set = create_train_set(INPUT_DIM, 100)

    cov_matrix = np.cov(train_set, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    largest_eig_val = np.max(eig_vals)
    largest_eig_vec = eig_vecs[:, np.argmax(eig_vals)]
    print(f'Largest eigenvalues: {largest_eig_val}, Largest eigenvector: {largest_eig_vec}')

    train(encoder, train_set, largest_eig_vec)
    plt.show()

    plt.plot(angle_list)
    plt.title('Angle between weight and eigenvector')
    plt.show()

    plt.plot(loss_list)
    plt.title('Loss')
    plt.show()
