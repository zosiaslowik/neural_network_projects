import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, expected_output):
        self.inputLayerNeurons, self.hiddenLayerNeurons, self.outputLayerNeurons = 2, 2, 1
        self.inputs = inputs
        self.expected_output = expected_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def train(self, activation, lr=0.1, epochs=10000, threshold=0.05, momentum=0.5):
        plt.figure(figsize=(12, 6))

        # Without Momentum
        plt.subplot(1, 2, 1)
        for i in range(5):
            np.random.seed(i)  # Added seed for reproducibility

            hidden_weights = np.random.uniform(size=(self.inputLayerNeurons, self.hiddenLayerNeurons))
            hidden_bias = np.random.uniform(size=(1, self.hiddenLayerNeurons))
            output_weights = np.random.uniform(size=(self.hiddenLayerNeurons, self.outputLayerNeurons))
            output_bias = np.random.uniform(size=(1, self.outputLayerNeurons))


            print("Initial hidden weights: ", hidden_weights)
            print("Initial hidden biases: ", hidden_bias)
            print("Initial output weights: ", output_weights)
            print("Initial output biases: ", output_bias)

            errors_list = []
            epochs_list = []
            i = 0
            mse = threshold + 0.05

            while i < epochs:
                # if mse < threshold:
                #     break

                # Forward Propagation
                epochs_list.append(i + 1)
                i += 1
                hidden_layer_activation = np.dot(self.inputs, hidden_weights)
                hidden_layer_activation += hidden_bias
                hidden_layer_output = self.tanh(hidden_layer_activation) if activation == 'tanh' else self.sigmoid(
                    hidden_layer_activation)

                output_layer_activation = np.dot(hidden_layer_output, output_weights)
                output_layer_activation += output_bias
                predicted_output = self.tanh(output_layer_activation) if activation == 'tanh' else self.sigmoid(
                    output_layer_activation)

                # Backpropagation
                error = self.expected_output - predicted_output
                mse = np.mean(error ** 2)
                errors_list.append(mse)
                d_predicted_output = error * self.sigmoid_derivative(
                    predicted_output) if activation == "sigmoid" else error * self.tanh_derivative(
                    predicted_output)

                error_hidden_layer = d_predicted_output.dot(output_weights.T)
                d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(
                    hidden_layer_output) if activation == "sigmoid" else error_hidden_layer * self.tanh_derivative(
                    hidden_layer_output)

                # Updating Weights and Biases
                output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
                output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
                hidden_weights += self.inputs.T.dot(d_hidden_layer) * lr
                hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

            plt.plot(epochs_list, errors_list)
            print("Final hidden weights: ", end='')
            print(*hidden_weights)
            print("Final hidden bias: ", end='')
            print(*hidden_bias)
            print("Final output weights: ", end='')
            print(*output_weights)
            print("Final output bias: ", end='')
            print(*output_bias)
            print("Output from neural network after ", i, " epochs: ", end='')
            print(*predicted_output)
        plt.title("Epochs vs error using " + activation + " (Without Momentum)")
        plt.xlabel("epochs")
        plt.ylabel("error")

        # With Momentum
        print("\n----------------------------------------------------------Z MOMENTUM----------------------------------------------------------\n")
        plt.subplot(1, 2, 2)
        for i in range(5):
            np.random.seed(i)  # Added seed for reproducibility

            m_hidden_weights = np.random.uniform(size=(self.inputLayerNeurons, self.hiddenLayerNeurons))
            m_hidden_bias = np.random.uniform(size=(1, self.hiddenLayerNeurons))
            m_output_weights = np.random.uniform(size=(self.hiddenLayerNeurons, self.outputLayerNeurons))
            m_output_bias = np.random.uniform(size=(1, self.outputLayerNeurons))

            v_m_output_weights = np.zeros_like(m_output_weights)
            v_m_output_bias = np.zeros_like(m_output_bias)
            v_m_hidden_weights = np.zeros_like(m_hidden_weights)
            v_m_hidden_bias = np.zeros_like(m_hidden_bias)

            print("\nInitial hidden weights: ", m_hidden_weights)
            print("Initial hidden biases: ", m_hidden_bias)
            print("Initial output weights: ", m_output_weights)
            print("Initial output biases: ", m_output_bias)

            errors_list = []
            epochs_list = []
            i = 0
            mse = threshold + 0.05

            while i < epochs:
                # if mse<threshold:
                #     break
                # Forward Propagation
                epochs_list.append(i + 1)
                i += 1
                m_hidden_layer_activation = np.dot(self.inputs, m_hidden_weights)
                m_hidden_layer_activation += m_hidden_bias
                m_hidden_layer_output = self.tanh(m_hidden_layer_activation) if activation == 'tanh' else self.sigmoid(
                    m_hidden_layer_activation)

                m_output_layer_activation = np.dot(m_hidden_layer_output, m_output_weights)
                m_output_layer_activation += m_output_bias
                m_predicted_output = self.tanh(m_output_layer_activation) if activation == 'tanh' else self.sigmoid(
                    m_output_layer_activation)

                # Backpropagation
                error = self.expected_output - m_predicted_output
                mse = np.mean(error ** 2)
                errors_list.append(mse)
                md_predicted_output = error * self.sigmoid_derivative(
                    m_predicted_output) if activation == "sigmoid" else error * self.tanh_derivative(
                    m_predicted_output)

                m_error_hidden_layer = md_predicted_output.dot(m_output_weights.T)
                md_hidden_layer = m_error_hidden_layer * self.sigmoid_derivative(
                    m_hidden_layer_output) if activation == "sigmoid" else m_error_hidden_layer * self.tanh_derivative(
                    m_hidden_layer_output)

                # Updating Weights and Biases with Momentum
                v_m_output_weights = momentum * v_m_output_weights + m_hidden_layer_output.T.dot(md_predicted_output) * lr
                m_output_weights += v_m_output_weights
                v_m_output_bias = momentum * v_m_output_bias + np.sum(md_predicted_output, axis=0, keepdims=True) * lr
                m_output_bias += v_m_output_bias

                v_m_hidden_weights = momentum * v_m_hidden_weights + self.inputs.T.dot(md_hidden_layer) * lr
                m_hidden_weights += v_m_hidden_weights
                v_m_hidden_bias = momentum * v_m_hidden_bias + np.sum(md_hidden_layer, axis=0, keepdims=True) * lr
                m_hidden_bias += v_m_hidden_bias

            plt.plot(epochs_list, errors_list)
            print("Final hidden weights: ", end='')
            print(*m_hidden_weights)
            print("Final hidden bias: ", end='')
            print(*m_hidden_bias)
            print("Final output weights: ", end='')
            print(*m_output_weights)
            print("Final output bias: ", end='')
            print(*m_output_bias)
            print("Output from neural network after ", i, " epochs: ", end='')
            print(*predicted_output)
        plt.title("Epochs vs error using " + activation + " (With Momentum)")
        plt.xlabel("epochs")
        plt.ylabel("error")

        plt.tight_layout()
        plt.show()

# Test the neural network with sigmoid activation
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(inputs, expected_output)

print("\nDLA SIGMOIDY: ")
nn.train(activation='sigmoid', threshold=0.005, epochs=10000, lr=0.4, momentum=0.9)

print("\nDLA TANGENSA HIPERBOLICZNEGO: ")
nn.train(activation='tanh', threshold=0.005, epochs=10000, lr=0.4, momentum=0.9)
