import os
import numpy as np
from numpy import log as ln

P=[1,1,1]

def tanh(x):
    tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return tanh



def forward(P, V,W):
    x0 = P[0]*V[0][0]+ P[1]*V[0][1]+P[2]*V[0][2]
    x1 = P[0]*V[1][0]+ P[1]*V[1][1]+P[2]*V[1][2]
    x2 = P[0]*V[2][0]+ P[1]*V[2][1]+P[2]*V[2][2]

    b0=1
    b1 = tanh(x1)
    b2 = tanh(x2)

    z= b0*W[0]+b1*W[1]+b2*W[2]
    a = tanh(z)
    return a, b0, b1, b2, z


def backprop(a, V, W, z, b0, b1, b2):
    et_a = -(-1 - a)

    da_dw0 = (1 - tanh(z) ** 2) * b0
    da_dw1 = (1 - tanh(z) ** 2) * b1
    da_dw2 = (1 - tanh(z) ** 2) * b2




    db1_dv11 = (1 - tanh(-2 * ln(2) + ln(4 / 3)) ** 2) * P[1]
    da_db1 = ((1 - tanh(z) ** 2) * W[1])

    db1_dv12 = (1 - tanh(-2 * ln(2) + ln(4 / 3)) ** 2) * P[2]
    da_db2 = ((1 - tanh(z) ** 2) * W[2])

    db1_dv10 = (1 - tanh(-2 * ln(2) + ln(4 / 3)) ** 2) * P[0]
    da_db0 = ((1 - tanh(z) ** 2) * W[0])


    V[1][1] = V[1][1] - et_a * db1_dv11 * da_db1
    V[1][2] = V[1][2] - et_a * db1_dv12 * da_db2
    V[1][0] = V[1][0] - et_a * db1_dv10 * da_db0
    W[1] = W[1] - et_a * da_dw1
    W[0] = W[0] - et_a * da_dw0
    W[2] = W[2] - et_a * da_dw2
    return V, W



V = [[0,0,0],
    [ln(4/3),-ln(2),-ln(2)],
     [ln(2), -ln(2),-ln(2)]]

W = [ln(2)+1/2, 5/8, 0]

a=100
i=0
while a!=-1 and i<5:
    print (V)
    print (W)
    a, b0, b1, b2, z = forward(P,V,W)
    #print (b0, b1, b2)
    print (a)
    V, W = backprop(a, V, W, z, b0, b1, b2)
    i+=1









