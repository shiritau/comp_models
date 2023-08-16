import os
import random
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


def onclick(encoder, input_dim):
    x_list = []
    global idx
    # global encoder
    for iter in range(1):
        random_idx = np.random.randint(0, 3)
        if random_idx == 0:
            X = np.random.normal((-10, 10, -10), 1, input_dim)
        if random_idx == 1:
            X = np.random.normal((-10, -10, -10), 1, input_dim)
        if random_idx == 2:
            X = np.random.normal((20, 0, 20), 1, input_dim)
        x_list.append(X)
        encoder.forward(X)
        encoder.backward(X)
        X_hat = np.dot(encoder.y, encoder.weights)
        loss = loss_fxn(X, X_hat)
        loss_list.append(loss)

        # Clear the previous plot
        ax.cla()
        ax.set_xlim(-20, 25)
        ax.set_ylim(-20, 25)
        ax.set_zlim(-20, 25)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Plot the weights
        ax.quiver(0, 0, 0, encoder.weights[0] * 10, encoder.weights[1] * 10, encoder.weights[2] * 10, color='r',
                  label='Weights')

        # Plot the input point
        ax.scatter(X[0], X[1], X[2], color='b', label='Input')

        # Plot the output point
        ax.scatter(X_hat[0], X_hat[1], X_hat[2], color='g', label='Output')

        ax.legend(['Weight', 'Input', 'Output'])

        # Clear existing annotations
        ax.texts.clear()

        # Add new annotations for weight, input, and output
        ax.text(10, 15, 20, f'iter: {idx + 1}', transform=ax.transAxes)
        ax.text(10, 15, 0, f'Input: {np.round(X, decimals=3)}', transform=ax.transAxes)
        ax.text(10, 15, -20, f'Weight: {np.round(encoder.weights, decimals=3)}', transform=ax.transAxes)
        ax.text(10, 15, -40, f'cluster {random_idx + 1}', transform=ax.transAxes)

        idx += 1
        fig.canvas.draw()


if __name__ == "__main__":
    INPUT_DIM = 3
    HIDDEN_DIM = 1
    LR = 0.001
    idx = 0
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

    # Connect the onclick event to the figure
    button = plt.Button(plt.axes([0.7, 0.9, 0.2, 0.075]), 'Next step')
    button.on_clicked(lambda event: onclick(encoder, INPUT_DIM))
    plt.show()

    plt.plot(loss_list)
    plt.title('Loss')
    plt.show()
