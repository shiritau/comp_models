import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights and biases
        self.weights = np.random.randn(input_dim, hidden_dim)


    def forward(self, input_data):
        self.hidden_layer = np.dot(input_data, self.weights)
        self.output_layer = np.dot(self.weights, self.hidden_layer)

        return self.output_layer

    def backward(self, input_data, output_layer, learning_rate):
        output_error = input_data.reshape((2, 1)) -  output_layer


        self.weights -= (learning_rate * np.dot(self.hidden_layer,output_error.reshape((1, 2)))).T
        return output_error

    def train(self, num_epochs, learning_rate):


        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        weight_line, = ax.plot([0, self.weights[0]], [0, self.weights[1]], 'b', label='Weight')
        input_scatter = ax.scatter([], [], c='r', label='Input')
        output_scatter = ax.scatter([], [], c='g', label='Output')
        ax.legend(['Weight', 'Input', 'Output'])
        loss_list = []

        def onclick(event):

            if event.xdata is not None and event.ydata is not None:


                for iter in range(20):
                    case = 0
                    for iter in range(20):
                        if case == 0:
                            random_idx = np.random.randint(0, 2)
                            if random_idx == 0:
                                data = np.random.normal((-0.5, 0), 0.1, 2)
                            elif random_idx == 1:
                                data = np.random.normal((0.5, 0), 0.1, 2)
                        elif case == 1:
                            random_idx = np.random.randint(0, 4)
                            data = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
                            data = np.array(data)
                            data = data[random_idx]
                        elif case == 2:
                            x = np.random.uniform(-0.3, 0.3)
                            y = x + np.random.normal(0, 0.05)
                            data = [x, y]
                            data = np.array(data)
                        elif case == 3:
                            y = np.random.uniform(-0.8, 0.8)
                            x = np.random.normal(0, 0.05)
                            data = [x, y]
                            data = np.array(data)


                    data = data.reshape((1, 2))
                    output = self.forward(data)
                    loss = self.backward(data, output, learning_rate)

                    loss_list.append(np.sum(np.abs(loss)))
                    weight_line.set_data([0, self.weights[0]], [0, self.weights[1]])
                    data = data.reshape((1, -1))
                    input_scatter.set_offsets(data)
                    #reshape output to 1 rank
                    output = output.reshape((1, -1))
                    output_scatter.set_offsets(output[0])
                    output_scatter.set_array(output[0])

                    # Clear existing annotations
                    ax.texts.clear()

                    # Add new annotations for weight, input, and output
                    ax.text(0.02, 0.95, f'Input: {data}', transform=ax.transAxes)
                    ax.text(0.02, 0.85, f'Weight: {np.round(self.weights, decimals=3)}', transform=ax.transAxes)
                    #ax.text(0.02, 0.85, f'y: {np.round(y, decimals=3)}', transform=ax.transAxes)
                    ax.text(0.02, 0.8, f'Output: {np.round(output[0], decimals=3)}', transform=ax.transAxes)

                    fig.canvas.draw()


        # Connect the onclick event to the figure
        fig.canvas.mpl_connect('button_press_event', onclick)

        # Show the plot
        plt.show()

        plt.plot(loss_list)
        plt.show()



if __name__ == "__main__":

    autoencoder = Autoencoder(2, 1)
    autoencoder.train(1000, 0.1)




