import numpy as np
from numpy import log as ln



def tanh(x):
    tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return tanh



def forward(P, W0, W1, activation):
    h0_0 = W0[0][0]*P[0]+ W0[0][1]*P[1]
    h0_1 = W0[1][0]*P[0]+ W0[1][1]*P[1]

    s1_0 = activation(h0_0)
    s1_1 = activation(h0_1)

    h1= W1[0]*s1_0+W1[1]*s1_1
    y = activation(h1)

    return y,h0_1, h0_0, s1_0, s1_1, h1


def backprop(y_true, y_pred, P, W0, W1, h0_1, h0_0, s1_0, s1_1, h1):
    et_a = -(y_true - y_pred)

    da_dw1_00 = (1 - tanh(h1) ** 2) * s1_0
    da_dw1_01 = (1 - tanh(h1) ** 2) * s1_1


    ds1_0_dw0_00 = (1 - tanh(h0_0) ** 2) * P[0]
    ds1_0_dw0_01 = (1 - tanh(h0_0) ** 2) * P[1]
    da_ds1_0 = ((1 - tanh(h0_0) ** 2) * W1[0])

    ds1_1_dw0_10 = (1 - tanh(h0_1) ** 2) * P[0]
    ds1_1_dw0_11 = (1 - tanh(h0_1) ** 2) * P[1]
    da_ds1_1 = ((1 - tanh(h0_1) ** 2) * W1[1])

    W1[0] = W1[0] - et_a * da_dw1_00
    W1[1] = W1[1] - et_a * da_dw1_01
    W0[0][0] = W0[0][0] - et_a * da_ds1_0 * ds1_0_dw0_00
    W0[0][1] = W0[0][1] - et_a * da_ds1_0 * ds1_0_dw0_01
    W0[1][0] = W0[1][0] - et_a * da_ds1_1 * ds1_1_dw0_10
    W0[1][1] = W0[1][1] - et_a * da_ds1_1 * ds1_1_dw0_11
    return W0, W1



P = np.random.uniform(-3,3,2)
N_MIDDLE=100
W0 = np.random.uniform(0,1,(2,2))
W1 = np.random.uniform(0,1,(2,1))


# a=100
# i=0
# while a!=-1 and i<5:
#     print (V)
#     print (W)
#     a, b0, b1, b2, z = forward(P,V,W)
#     #print (b0, b1, b2)
#     print (a)
#     V, W = backprop(a, V, W, z, b0, b1, b2)
#     i+=1









