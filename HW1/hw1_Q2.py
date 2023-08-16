import numpy as np
import matplotlib.pyplot as plt


def PLA(x, y, lr=0.1):
    W = np.array([0, 0]).astype('float64')
    done = False
    iteration = 1
    correct_list = []
    while done == False and iteration < 100:
        correct = 0
        # beginning of iteration
        for x_u, y_0 in zip(x, y):
            x_u = np.array(x_u)
            y_u = np.sign(np.dot(np.transpose(W), x_u))
            if y_0 == y_u:
                correct += 1
                continue
            else:
                # update weights
                W += lr * (y_0 * x_u)

        # at end of each iteration
        correct_list.append(correct)
        if correct == len(x):
            done = True
        iteration += 1

    if done == False and iteration == 100:
        return 0
    else:
        return iteration


if __name__ == '__main__':
    x1_mean = [3,-3]
    x2_mean = [-3,3]
    sigma = [0.5,1,2,3]

    success_rate_per_sigma = []
    avg_iter_per_sigma = []
    for s in sigma:
        success = []
        iterations = []
        for n in range(100):
            std = [s ** 2, s ** 2]
            # sample 10 random points from each distribution
            x1_samples = np.random.normal(x1_mean, std, size=(10, 2))
            x2_samples = np.random.normal(x2_mean, std, size=(10, 2))
            x = np.concatenate((x1_samples, x2_samples))
            y = np.concatenate((np.ones((len(x1_samples), 1)), np.ones((len(x2_samples), 1)) * -1)).astype('float64')
            result = PLA(x, y)
            # result 0 means PLA did not end successfully, and no separation line was found
            # otherwise, the result value is the iteration number on which PLA was successful
            if result == 0:
                success.append(result)
            else:
                success.append(1)
                iterations.append(result)

        # after finished 100 runs, add percentage of successful trials out of 100 for each sigma
        success_rate_per_sigma.append(np.sum(success) / 100)

        # if at least one of the 100 runs completed successfully, get the mean # iterations
        if iterations != []:
            avg_iter_per_sigma.append(np.mean(iterations))
        else:
            avg_iter_per_sigma.append(0)

    plt.plot(sigma, success_rate_per_sigma)
    plt.title('Success rate per sigma')
    plt.xlabel('Sigma')
    plt.ylabel('Sucess rate')
    plt.show()

    plt.plot(sigma, avg_iter_per_sigma)
    plt.title('Average iterations required per sigma, n=100')
    plt.xlabel('Sigma')
    plt.ylabel('# Iterations')
    plt.show()


    # Experimenting with lr
    success_per_lr = []
    avg_iter_per_lr = []
    lrs = np.linspace(0.1, 1, 20)
    sigma = 1.5
    for lr in lrs:
        success = []
        iterations = []
        for n in range(100):
            x1_samples = np.random.normal(x1_mean, sigma ** 2, size=(20, 2))
            x2_samples = np.random.normal(x2_mean, sigma ** 2, size=(20, 2))
            x = np.concatenate((x1_samples, x2_samples))
            y = np.concatenate((np.ones((len(x1_samples), 1)), np.ones((len(x2_samples), 1)) * -1)).astype('float64')
            result = PLA(x, y, lr)
            if result == 0:
                success.append(result)
            else:
                success.append(1)
                iterations.append(result)
        success_per_lr.append(np.sum(success) / 100)
        if iterations != []:
            avg_iter_per_lr.append(np.mean(iterations))
        else:
            avg_iter_per_lr.append(0)

    plt.plot(lrs, success_per_lr)
    plt.title(f'learning rate vs. success rate, sigma = {sigma}')
    plt.xlabel('learning rate')
    plt.ylabel('Sucess rate')
    plt.show()
    plt.plot(lrs, avg_iter_per_lr)
    plt.title(f'learning rate vs. # iteration, sigma = {sigma}')
    plt.xlabel('learning rate')
    plt.ylabel('# Iterations')
    plt.show()

    # Visualizing the data
    x1_mean = [3,-3]
    x2_mean = [-3,3]
    sigma = 1.5
    x1_samples = np.random.normal(x1_mean, sigma**2 , size = (20,2))
    x2_samples = np.random.normal(x2_mean, sigma**2 ,size = (20,2))
    x = np.concatenate((x1_samples, x2_samples))
    y = np.concatenate((np.ones((len(x1_samples),1)), np.ones((len(x2_samples),1))*-1)).astype('float64')
    #plot
    x1 = [i[0] for i in x]
    x2 = [i[1] for i in x]
    colors = ['red' if label == -1 else 'blue' for label in y]
    plt.scatter(x1, x2, c=colors)
