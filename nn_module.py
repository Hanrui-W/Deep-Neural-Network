import numpy as np
from matplotlib import pyplot
import sys


# NeuralNetwork class
class NeuralNetwork:
    # Define the class constructor
    def __init__(self, x_train, y_train, layers, learning_rate=0.1, max_iter=1000):
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers
        self.parameters = self.init_parameters()
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    # Initialize network parameters
    def init_parameters(self):
        parameter = {}
        for i in range(1, len(self.layers)):
            parameter['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(self.layers[i], 1)
            assert (parameter['W' + str(i)].shape == (self.layers[i], self.layers[i - 1])) * 0.01
            assert (parameter['b' + str(i)].shape == (self.layers[i], 1))
        return parameter

    # Define the ReLU activation function
    @staticmethod
    def relu(z):
        a = np.maximum(0, z)
        return a, z

    # Define the sigmoid activation function
    @staticmethod
    def sigmoid(z):
        a = 1 / (1 + np.exp(-z))
        return a, z

    # Initialize the forward propagation for one layer 'l'.
    # Input the activation from the previous layer and the weights,
    # output Z_l and stores excessive parameter in linear_cache.
    def linear_forward(self, A_prev, layer):
        W = self.parameters['W' + str(layer)]
        b = self.parameters['b' + str(layer)]
        Z = np.dot(W, A_prev) + b
        assert (Z.shape == (W.shape[0], A_prev.shape[1]))
        linear_cache = (A_prev, W, b)
        return Z, linear_cache

    # Initialize the forward prop for one activation layer.
    @staticmethod
    def activation_forward(Z, activation_type):
        A = None
        if activation_type == 'relu':
            A, Z = NeuralNetwork.relu(Z)
            assert (A.shape == Z.shape)
            pass
        elif activation_type == 'sigmoid':
            A, Z = NeuralNetwork.sigmoid(Z)
            assert (A.shape == Z.shape)
            pass
        activation_cache = Z
        return A, activation_cache

    # Initialize forward prop for the entire network
    def forward_propagation(self):
        caches = []
        A = self.x_train
        for i in range(1, len(self.parameters) // 2):
            Z, linear_cache = self.linear_forward(A, i)
            A, activation_cache = NeuralNetwork.activation_forward(Z, 'relu')
            cache = (linear_cache, activation_cache)
            caches.append(cache)
            pass
        L = len(self.parameters) // 2
        Z, linear_cache = self.linear_forward(A, L)
        A, activation_cache = NeuralNetwork.activation_forward(Z, 'sigmoid')
        cache = (linear_cache, activation_cache)
        caches.append(cache)

        assert (A.shape == self.y_train.shape)

        return A, caches

    # Compute the loss
    def get_loss(self, A_L):
        assert (A_L.shape == self.y_train.shape)
        m = A_L.shape[1]
        J = - np.sum((self.y_train * np.log(A_L) + (1 - self.y_train) * np.log(1 - A_L))) / m
        return J

    # Get the derivative of the sigmoid function
    @staticmethod
    def d_sigmoid(z):
        a, z = NeuralNetwork.sigmoid(z)
        return z * (1 - z)
        pass

    # Get the derivative of the ReLU function
    @staticmethod
    def d_relu(z):
        compare = (z >= 0)
        return compare

    # Compute the derivatives of the neural network
    def linear_backward(self, dA_l, A_l_prev, activation_function, Z_l, W_l):
        dZ_l = None
        m = self.y_train.shape[1]
        if activation_function == 'relu':
            dZ_l = dA_l * NeuralNetwork.d_relu(Z_l)
        elif activation_function == 'sigmoid':
            dZ_l = dA_l * NeuralNetwork.d_sigmoid(Z_l)
        dw_l = np.dot(dZ_l, A_l_prev.T) / m
        db_l = np.sum(dZ_l, axis=1, keepdims=True) / m
        da_prev = np.dot(W_l.T, dZ_l)
        return dw_l, db_l, da_prev

    # Kick start the training
    def init_training(self):
        gradients = []

        m = self.y_train.shape[1]

        A, caches = self.forward_propagation()

        # The last layer is the sigmoid layer
        d_Z_last = A - self.y_train
        linear_cache, activation_cache = caches[-1]
        A_prev, W_l, b_l = linear_cache

        # Z_l = activation_cache

        dw_l_last = np.dot(d_Z_last, A_prev.T) / m
        db_last = np.sum(d_Z_last, axis=1, keepdims=True) / m
        d_A_prev = np.dot(W_l.T, d_Z_last)
        gradients.append((dw_l_last, db_last))

        for i in reversed(range(len(caches) - 1)):
            d_A_l = d_A_prev
            linear_cache, activation_cache = caches[i]
            A_prev, W_l, b_l = linear_cache
            Z_l = activation_cache
            dw_l, db_l, d_A_prev = self.linear_backward(dA_l=d_A_l, A_l_prev=A_prev,
                                                        activation_function='relu', Z_l=Z_l,
                                                        W_l=W_l)
            gradients.append((dw_l, db_l))
            pass
        return gradients

    def train(self, display=True, classifier=True):
        costs = []
        for i in range(0, self.max_iter + 1):
            if i % 100 == 0 and display:
                loss = self.get_cost()
                print('loss is: ' + str(loss) + ' | ' + 'iteration ' + str(i))
                costs.append(loss)

                print("accuracy is " + str(self.test_eval(classifier)) + ".")
            gradients = self.init_training()
            gradients.reverse()
            for j in reversed(range(len(self.parameters) // 2)):
                dw_l, db_l = gradients[j]
                self.parameters['W' + str(j + 1)] = self.parameters['W' + str(j + 1)] - self.learning_rate * dw_l
                self.parameters['b' + str(j + 1)] = self.parameters['b' + str(j + 1)] - self.learning_rate * db_l
        return costs, self.parameters

    def test_eval(self, classifier):
        A, caches = self.forward_propagation()

        if classifier:
            compare = (A > 0.5)
            result = (compare == self.y_train)
            score = np.sum(result) / self.x_train.shape[1]
            return score
        else:
            raw_score = 0
            for i in range(A.shape[1]):
                if A[:, i].argmax() == self.y_train[:, i].argmax():
                    raw_score += 1
                    pass
                pass
            return raw_score / A.shape[1]

    def get_cost(self):
        A, caches = self.forward_propagation()
        loss = self.get_loss(A)
        return loss

    def query(self, inputs):
        A = inputs
        for i in range(1, len(self.parameters) // 2):
            Z, linear_cache = self.linear_forward(A, i)
            A, activation_cache = NeuralNetwork.activation_forward(Z, 'relu')

        L = len(self.parameters) // 2
        Z, linear_cache = self.linear_forward(A, L)
        A, activation_cache = NeuralNetwork.activation_forward(Z, 'sigmoid')
        return A


def main():
    print("Opening training set and converting to np array, please wait:")
    x_train = np.genfromtxt("mnist_dataset/mnist_train.csv", delimiter=',')
    x_train = np.transpose(x_train)

    print("Opening labeled set, please wait:")
    labeled_set = x_train[0:1][:]
    print("Shape of the labeled set is " + str(labeled_set.shape))

    print("Scaling the training set:")
    x_train = x_train[1:][:] / 255.0 * 0.99 + 0.01
    print("Shape of the training set is " + str(x_train.shape))

    print("creating new labeled set:")
    y_train = np.zeros((10, labeled_set.shape[1])) + 0.01
    for i in range(labeled_set.shape[1]):
        index = int(labeled_set.item((0, i)))
        y_train[index][i] = 0.99
    print("The shape of the new labeled set is: " + str(y_train.shape))

    neuralNet = NeuralNetwork(x_train=x_train, y_train=y_train, layers=(x_train.shape[0], 100, 100, 10),
                              learning_rate=0.1, max_iter=2500)

    neuralNet.train(display=True, classifier=False)

    print("Opening test set and converting to np array, please wait...")
    x_test = np.genfromtxt("mnist_dataset/mnist_test.csv", delimiter=',')

    y_test = x_test[:, 0:1]
    x_test = x_test[:, 1:]

    x_test = np.transpose(x_test)
    y_test = np.transpose(y_test)

    print("The shape of the test set is " + str(x_test.shape))
    print("The shape of the test labels are " + str(y_test.shape))

    for i in range(x_test.shape[1]):
        # Reshape array into 28 * 28 matrix
        image_array = np.asfarray(x_test[:, i:i+1]).reshape([28, 28])

        # Change the range of the input to 0.01 to 1
        output_list = neuralNet.query((np.asfarray(x_test[:, i:i+1]) / 255.0 * 0.99) + 0.01)

        max_index = output_list.argmax()
        print("The neural network determines that this image is the digit ", max_index)
        print("The actual digit is " + str(y_test.item(0, i)))

        pyplot.imshow(image_array, cmap='Greys', interpolation='None')
        pyplot.show()

        text = input("Press enter to continue. Type 'exit' to exit.")

        if text == 'exit':
            sys.exit()


if __name__ == "__main__":
    main()
