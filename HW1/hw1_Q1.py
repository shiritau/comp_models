import numpy as np
import matplotlib.pyplot as plt

def PLA_ordered(x, y, W):
  order = [1,5,2,6,3,7,4,8]
  done = False
  iteration = 1
  while done == False and iteration <100:
    print (f'Iteration: {iteration}')
    correct = 0
    for i, idx in enumerate(order, 1):
      x_u = np.array(x[idx-1])
      y_0 = y[idx-1]
      y_u = np.sign(np.dot(np.transpose(W), x_u))
      print (f'{i}: x_u: {x_u}, W: {W} y_0: {y_0}, y_u: {y_u}')
      if y_0 == y_u:
        correct+=1
        continue
      else:
        W+=y_0*x_u
        print (f'new W: {W}')
    if correct ==len(x):
      done = True
    iteration +=1


def PLA_ordered_with_lr(x, y, W, lr=0.01):
  order = [1,5,2,6,3,7,4,8]
  done = False
  iteration = 1
  while done == False and iteration <100:
    print (f'Iteration: {iteration}')
    correct = 0
    for i, idx in enumerate(order, 1):
      x_u = np.array(x[idx-1])
      y_0 = y[idx-1]
      y_u = np.sign(np.dot(np.transpose(W), x_u))
      print (f'{i}: x_u: {x_u}, W: {W} y_0: {y_0}, y_u: {y_u}')
      if y_0 == y_u:
        correct+=1
        continue
      else:
        W+=lr* y_0*x_u
        print (f'new W: {W}')
    if correct ==len(x):
      done = True
    iteration +=1


if __name__ == '__main__':
    # data examination
    x_train = np.array([(-1, 4), (-2, 1), (-1, 2), (-4, 1), (2, -1), (1, -2), (1, -4), (4, -1)])
    y_train = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # plot
    x1 = [i[0] for i in x_train]
    x2 = [i[1] for i in x_train]
    colors = ['red' if label == -1 else 'blue' for label in y_train]
    plt.scatter(x1, x2, c=colors)
    plt.show()

    # PLA
    W = np.array([0, 0]).astype('float64')
    PLA_ordered(x_train, y_train, W)

    # try PLA with learning rate
    W = np.array([0, 0]).astype('float64')
    PLA_ordered_with_lr(x_train, y_train, W)



