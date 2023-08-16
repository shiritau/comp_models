import os
import random
import numpy as np
import matplotlib.pyplot as plt




def encoder(X, W):
    y = X[0]*W[0] + X[1]*W[1]
    return y

def decoder(y, W):
    X_hat = np.zeros(2)
    X_hat[0] = y*W[0]
    X_hat[1] = y*W[1]
    return X_hat



def loss_fxn(X, X_hat):
    return np.sum(np.abs(0.5*((X-X_hat)**2)))

def train(X, W, lr):
    y = encoder(X, W)
    X_hat = decoder(y, W)
    loss = loss_fxn(X, X_hat)
    grad = np.dot((X-X_hat),y)
    W[0] = W[0] + lr*grad[0]
    W[1] = W[1] + lr*grad[1]
    return X_hat, W, y, loss




fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')

# init random 2x2 weight vector between -1 and 1
W = np.random.uniform(-1, 1, 2)


idx=0
weight_line, = ax.plot([0, W[0]], [0, W[1]], 'b', label='Weight')
input_scatter = ax.scatter([], [], c='r', label='Input')
output_scatter = ax.scatter([], [], c='g', label='Output')
ax.legend(['Weight', 'Input', 'Output'])
loss_list = []
def onclick(event):

    if event.xdata is not None and event.ydata is not None:
       # X_list.append([event.xdata, event.ydata])
        global W
        for iter in range(20):
            random_idx = np.random.randint(0, 2)
            if random_idx == 0:
                data = np.random.normal((-0.5, 0), 0.1, 2)
            elif random_idx == 1:
                data = np.random.normal((0.5, 0), 0.1, 2)
            X_hat, W, y, loss= train(data, W, 0.1)
            loss_list.append(loss)
            weight_line.set_data([0, W[0]], [0, W[1]])
            input_scatter.set_offsets(data)
            output_scatter.set_offsets(X_hat)
            output_scatter.set_array(X_hat)


           # Clear existing annotations
            ax.texts.clear()

            # Add new annotations for weight, input, and output
            ax.text(0.02, 0.95, f'Input: {data}', transform=ax.transAxes)
            ax.text(0.02, 0.9, f'Weight: {np.round(W, decimals=3)}', transform=ax.transAxes)
            ax.text(0.02, 0.85, f'y: {np.round(y, decimals=3)}', transform=ax.transAxes)
            ax.text(0.02, 0.8, f'Output: {np.round(X_hat,decimals=3)}', transform=ax.transAxes)

            fig.canvas.draw()

# Connect the onclick event to the figure
fig.canvas.mpl_connect('button_press_event', onclick)

# Show the plot
plt.show()

plt.plot(loss_list)
plt.show()