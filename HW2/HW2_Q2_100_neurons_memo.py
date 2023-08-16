import numpy as np
from numpy import log as ln
from scipy import signal
import matplotlib.pyplot as plt
import time


def peaks(X,Y):
    Z = 3*(1-X)**2*np.exp(-(X**2) - (Y+1)**2) \
       - 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2) \
       - 1/3*np.exp(-(X+1)**2 - Y**2)
    return Z

def new_peaks(X,Y, W0, W1):
    new_Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            P = [X[i][i], Y[j][j]]
            y, s1, h1, h2 = forward(P, W0, W1)
            new_Z[i][j] = y
    return new_Z
def tanh(x):
    tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return tanh


def forward(P, W0, W1, N_middle):
    b = 0.1 # bias
    h1=[]
    s1=[]
    for i in range( N_middle):
        h1_i= W0[i][2]*b
        for j in range(2):
            h1_i+=W0[i][j]*P[j]
        h1.append(h1_i)
        s1.append(tanh(h1_i))

    h2 = 0
    for i in range( N_middle):
        h2+=W1[0][i]*s1[i]

    y = tanh(h2)

    return y, s1, h1, h2, b

def backprop(y_true, y_pred, P, W0, W1, N_middle, s1, h1, h2, b, lr=0.001):
    et_a = -(y_true - y_pred)

    # da_dw1=[]
    # ds1_dw0=[]
    # da_ds1=[]
    # for i in range( N_middle):
    #     da_dw1.append((1 - tanh(h2) ** 2) * s1[i])
    #
    #     ds1_i_dw0_i0 = (1 - tanh(h1[i]) ** 2) * P[0]
    #     ds1_i_dw0_i1 = (1 - tanh(h1[i]) ** 2) * P[1]
    #     ds1_i_dw0_i2 = (1 - tanh(h1[i]) ** 2) * b
    #     da_ds1_i = ((1 - tanh(h2) ** 2) * W1[0][i])
    #     ds1_dw0.append([ds1_i_dw0_i0, ds1_i_dw0_i1, ds1_i_dw0_i2])
    #     da_ds1.append(da_ds1_i)


    for i in range(N_middle):
        W1[0][i] = W1[0][i] - lr* et_a *((1 - tanh(h2) ** 2) * s1[i])
        W0[i][0] = W0[i][0] - lr*et_a *((1 - tanh(h2) ** 2) * W1[0][i]) * (1 - tanh(h1[i]) ** 2) * P[0]
        W0[i][1] = W0[i][1] - lr*et_a *((1 - tanh(h2) ** 2) * W1[0][i]) * (1 - tanh(h1[i]) ** 2) * P[1]
        W0[i][2] = W0[i][2] - lr*et_a *((1 - tanh(h2) ** 2) * W1[0][i]) * (1 - tanh(h1[i]) ** 2) * b

    return W0, W1


def plot_peaks(W0, W1, N_MIDDLE, density=100):
    z_pred = np.zeros((density, density))
    X = np.linspace(-3, 3, density)
    Y = np.linspace(-3, 3, density)
    X, Y = np.meshgrid(X, Y)
    Z = 0.1 * peaks(X, Y)
    for i in range(density):
        for j in range(density):
            P = [X[i][i], Y[j][j]]
            y, s1, h1, h2, b = forward(P, W0, W1, N_MIDDLE)
            z_pred[i][j] = y

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X, Y, z_pred, cmap='viridis', edgecolor='none')
    ax.title.set_text('Predicted')
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.title.set_text('True')
    plt.show()

def train(W0, W1, N_MIDDLE, num_iter=10000, density=20):
    loss = []
    smooth_loss = []
    start = time.time()
    for iter in range(num_iter):

        # Choose a random index in the meshgrid
        i = np.random.randint(0, density)
        j = np.random.randint(0, density)
        P = [X[i][i], Y[j][j]]
        y_true = Z[i][j]

        y, s1, h1, h2, b = forward(P, W0, W1, N_MIDDLE)
        W0, W1 = backprop(y_true, y, P, W0, W1, N_MIDDLE, s1, h1, h2,b)
        loss.append((y_true - y) ** 2)
        if iter % 1000 == 0:
            print('Time taken for 1000 iteration: {} is {:.5f} seconds'.format(iter, time.time() - start))
            start = time.time()
            smooth_loss.append(np.mean(loss[-1000:]))
        if iter % 1000 == 0:
            print('Iter: {}, Loss: {:.4f}'.format(iter, np.mean(loss)))
    return W0, W1, smooth_loss


if __name__ == '__main__':
    N_MIDDLE=100

    # Define the ranges for x and y variables
    density = 500
    X = np.linspace(-3, 3, density)
    Y = np.linspace(-3, 3, density)
    X, Y = np.meshgrid(X, Y)
    Z = 0.1* peaks(X, Y)

    #train 100 times, and show best loss
    best_loss = 10000
    best_loss_list=[]
    for i in range(5):
        W0 = np.random.uniform(-1, 1, (N_MIDDLE, 3))
        W1 = np.random.uniform(-1, 1, (1, N_MIDDLE))
        W0, W1, loss = train(W0, W1, N_MIDDLE, num_iter=100000, density=density)
        if loss[-1]<best_loss:
            best_loss=loss[-1]
            best_loss_list.append(best_loss)
            best_W0=W0.copy()
            best_W1=W1.copy()


    plt.plot(best_loss_list)
    plt.show()

    # define loss
    loss=0
    for k in range(1000):
        i = np.random.randint(0,density)
        j = np.random.randint(0,density)
        P = [X[i][i], Y[j][j]]
        y_true = Z[i][j]
        y_pred, s1, h1, h2,b = forward(P, best_W0,best_W1, N_MIDDLE)
        loss+=(y_true-y_pred)**2
    print (loss/1000)

    plot_peaks(best_W0, best_W1, N_MIDDLE, density=100)














