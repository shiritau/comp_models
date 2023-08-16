import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights and biases
        self.weights = np.random.randn(hidden_dim, input_dim)
        self.output_weights = np.random.randn(input_dim, hidden_dim)

    def forward(self, input_data):
        self.hidden_layer = np.dot(self.weights, input_data)
        self.output_layer = np.dot(self.output_weights, self.hidden_layer)


        return self.output_layer

    def backward(self, input_data, output_layer, learning_rate):
        output_error = output_layer - input_data
        hidden_error = np.dot(self.output_weights.T, output_error)

        self.output_weights -= learning_rate * np.dot(output_error, self.hidden_layer.T)
        self.weights -= learning_rate * np.dot(hidden_error, input_data.T)
        return output_error

    def train(self, input_data, num_epochs, learning_rate):
        for data in input_data:
            data = data.reshape((-1, 1))
            output = self.forward(data)
            self.backward(data, output, learning_rate)

        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')


        weight_line, = ax.plot([0, self.output_weights[0]], [0, self.output_weights[1]], 'b', label='Weight')
        input_scatter = ax.scatter(input_data[0][0], input_data[0][1], c='r', label='Input')
        output_scatter = ax.scatter([], [], c='g', label='Output')
        ax.legend(['Weight', 'Input', 'Output'])
        loss_list = []

        def onclick(event):

            if event.xdata is not None and event.ydata is not None:

                case = 1
                for iter in range(200):
                    if case == 0:
                        random_idx = np.random.randint(0, 2)
                        if random_idx == 0:
                            data = np.random.normal((-0.5,0), 0.1, 2)
                        elif random_idx == 1:
                            data = np.random.normal((0.5,0), 0.1, 2)
                    elif case == 1:
                        random_idx = np.random.randint(0, 4)
                        data = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
                        data=np.array(data)
                        data = data[random_idx]
                    elif case ==2:
                        x = np.random.uniform(-0.7, 0.7)
                        y = x + np.random.normal(0, 0.1)
                        data = [x, y]
                        data = np.array(data)


                    data = data.reshape((-1, 1))
                    output = self.forward(data)
                    loss = self.backward(data, output, learning_rate)

                    loss_list.append(np.sum(np.abs(loss)))
                    weight_line.set_data([0, self.output_weights[0]], [0, self.output_weights[1]])
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
                    ax.text(0.02, 0.85, f'Weight: {np.round(self.output_weights, decimals=3)}', transform=ax.transAxes)
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
    input_data = np.array([[-0.5,0],[0.45,0.1],[-0.52,-0.05],[0.35,0.1],[-0.4,-0.1],[0.5,0.2],[-0.51,0.03],[0.45,0.17],[-0.52,-0.05],[0.35,-0.14]])
    idx = 0

    autoencoder = Autoencoder(2, 1)
    autoencoder.train(input_data, 1000, 0.01)




