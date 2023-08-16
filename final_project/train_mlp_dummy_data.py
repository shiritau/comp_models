import numpy as np
from clearml import Task
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import logging


class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.lr = lr
        self.W0 = np.random.randn(self.hidden_size, self.input_size)
        self.W1 = np.random.randn(self.hidden_size, self.hidden_size)
        self.W2 = np.random.randn(self.output_size, self.hidden_size)

    def tanh(self, x):
        tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return tanh

    def sigmoid(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, data):
        self.h1 = []
        self.s1 = []
        self.h2 = []
        self.s2 = []

        # layer 1
        for i in range(self.hidden_size):
            h1_i = 0
            for j in range(self.input_size):
                h1_i += self.W0[i][j] * data[j]

            self.h1.append(h1_i)
            self.s1.append(self.tanh(h1_i))

        # layer 2
        for i in range(self.hidden_size):
            h2_i = 0
            for j in range(self.hidden_size):
                h2_i += self.W1[i][j] * self.s1[j]
            self.h2.append(h2_i)
            self.s2.append(self.tanh(h2_i))

        # output layer
        self.h3 = 0
        for i in range(self.hidden_size):
            self.h3 += self.W2[0][i] * self.s2[i]

        y = self.sigmoid(self.h3)
        return y

    def bce_loss(self, y_true, y_pred):
        bce = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return bce

    def backprop(self, loss, data):
        for i in range(self.hidden_size):
            self.W2[0][i] = self.W2[0][i] - self.lr * loss * ((1 - self.tanh(self.h3) ** 2) * self.s2[i])
            for j in range(self.hidden_size):
                self.W1[i][j] = self.W1[i][j] - self.lr * loss * ((1 - self.tanh(self.h3) ** 2) * self.W2[0][i]) * (
                        1 - self.tanh(self.h2[i]) ** 2) * self.s1[j]
            for j in range(self.input_size):
                self.W0[i][j] = self.W0[i][j] - self.lr * loss * ((1 - self.tanh(self.h3) ** 2) * self.W1[0][i]) * (
                        1 - self.tanh(self.h1[i]) ** 2) * data[j]

        return self.W0, self.W1


def calc_sens_spec(output, label, accuracy_threshold):
    eps = 1e-7
    tp = ((output > accuracy_threshold) & (label == 1)).sum().item()
    tn = ((output < accuracy_threshold) & (label == 0)).sum().item()
    fp = ((output > accuracy_threshold) & (label == 0)).sum().item()
    fn = ((output < accuracy_threshold) & (label == 1)).sum().item()
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return sensitivity, specificity


def train(X, Y, model, num_epochs, weights_dir, exp_name, accuracy_threshold=0.5):
    writer = SummaryWriter(log_dir=weights_dir, comment=exp_name)

    logging.info('Start training')
    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_train_sens = []
        epoch_train_spec = []

        for batch_idx, (data, label) in enumerate(zip(X, Y)):

            output = model.forward(data)
            loss = model.bce_loss(label, output)
            model.backprop(loss, data)

            epoch_train_loss.append(loss.item())
            accuracy = ((output > accuracy_threshold) == label)  # .sum().item() / len(label)

            epoch_train_acc.append(accuracy)
            sensitivity, specificity = calc_sens_spec(output, label, accuracy_threshold)
            epoch_train_sens.append(sensitivity)
            epoch_train_spec.append(specificity)

            if batch_idx % 10 == 0:
                print('Train Epoch: {}, iteration: {}, Loss: {:.6f}, Acc: {:.6f}'.format(epoch, batch_idx,
                                                                                         np.mean(epoch_train_loss),
                                                                                         np.mean(epoch_train_acc)))
        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(np.array(epoch_train_acc)), epoch)
        writer.add_scalar('Sensitivity/train', np.mean(np.array(epoch_train_sens)), epoch)
        writer.add_scalar('Specificity/train', np.mean(np.array(epoch_train_spec)), epoch)


if __name__ == '__main__':
    ROOT_DIR = r"C:\Users\shiri\Documents\School\Master\Courses\Comp_models_of_learning\final_project\brats"
    WEIGHTS_DIR = f'{ROOT_DIR}/exps'
    DF_PATH = f'{ROOT_DIR}/paths.csv'
    NUM_EPOCHS = 150
    ACCURACY_THRESHOLD = 0.5

    mlp_model = MLP(hidden_size=10, input_size=2, output_size=1, lr=0.0001)
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    task = Task.init(project_name="tumour_detection", task_name=f"mlp_{current_time}")

    # create dummy data
    x1_mean = [3, -3]
    x2_mean = [-3, 3]
    sigma = 1.5
    x1_samples = np.random.normal(x1_mean, sigma ** 2, size=(200, 2))
    x2_samples = np.random.normal(x2_mean, sigma ** 2, size=(200, 2))
    x = np.concatenate((x1_samples, x2_samples))
    y = np.concatenate((np.ones((len(x1_samples), 1)), np.zeros((len(x2_samples), 1)))).astype('float64')

    # plot data
    x1 = [i[0] for i in x]
    x2 = [i[1] for i in x]
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(x1, x2, c=colors)
    plt.show()
    # shuffle y
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    train(x, y, mlp_model, NUM_EPOCHS, WEIGHTS_DIR, task.name, ACCURACY_THRESHOLD)
