import random
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class NeuralNetwork:
    def __init__(self):
        self.weights = None
        self.biases = None

    def init_parameters(self, input_neurons, hidden_neurons, output_neurons, epochs, lr, batch_size):
        # Initialize the neural network parameters
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.epochs = epochs
        self.learning_rate = lr
        self.batch_size = batch_size

        # Initialize weights and biases with random values
        self.biases = [np.random.randn(hidden_neurons, 1), np.random.randn(output_neurons, 1)]
        self.weights = [np.random.randn(hidden_neurons, input_neurons), np.random.randn(output_neurons, hidden_neurons)]

    def train(self, train_data, test_data):
        # Train the neural network using mini-batch gradient descent
        accuracies = []  # Define the variable "test_accuracies"
        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            random.shuffle(train_data)  # Shuffle the training data

            # Create mini-batches from the training data
            mini_batches = [train_data[i:i + self.batch_size] for i in range(0, len(train_data), self.batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            # Evaluate the model on the test data and print the accuracy
            accuracy = self.evaluate(test_data)
            end_time = time.time()
            accuracies.append(accuracy)
            print(f"Epoch #{epoch} \nAccuracy = {accuracy / 100}% \nTime Elapsed {round(end_time - start_time, 2)}s\n")
        print("Training complete")
        return accuracies

    def update_mini_batch(self, mini_batch):
        # Initialize the changes in biases and weights to zero
        bias_deltas = [np.zeros(b.shape) for b in self.biases]
        weight_deltas = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # Compute the gradient for the current mini-batch
            delta_b, delta_w = self.backpropagate(x, y)
            bias_deltas = [bd + d for bd, d in zip(bias_deltas, delta_b)]
            weight_deltas = [wd + d for wd, d in zip(weight_deltas, delta_w)]

        # Update the weights and biases using the gradients
        self.weights = [w - (self.learning_rate / len(mini_batch)) * wd for w, wd in zip(self.weights, weight_deltas)]
        self.biases = [b - (self.learning_rate / len(mini_batch)) * bd for b, bd in zip(self.biases, bias_deltas)]

    def backpropagate(self, x, y):
        # Perform the backpropagation algorithm to compute the gradients
        x = np.array(x).reshape((len(x), 1)) / 255.0  # Normalize input
        y = int(y)

        # Feedforward pass
        z1 = np.dot(self.weights[0], x) + self.biases[0]
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.weights[1], a1) + self.biases[1]
        a2 = self.sigmoid(z2)

        # Compute the output error
        y_vector = np.zeros((self.output_neurons, 1))
        y_vector[y] = 1.0
        output_error = (a2 - y_vector) * self.sigmoid_derivative(a2)

        # Compute the hidden layer error
        hidden_error = np.dot(self.weights[1].T, output_error) * self.sigmoid_derivative(a1)

        # Compute the gradient for biases and weights
        delta_biases = [hidden_error, output_error]
        delta_weights = [np.dot(hidden_error, x.T), np.dot(output_error, a1.T)]

        return delta_biases, delta_weights

    def evaluate(self, test_data):
        # Evaluate the model on the test data
        correct = 0
        for x, y in test_data:
            if self.predict(x) == y:
                correct += 1
        return correct

    def predict(self, x):
        # Predict the output for a given input
        x = np.array(x).reshape((len(x), 1)) / 255.0
        z1 = np.dot(self.weights[0], x) + self.biases[0]
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.weights[1], a1) + self.biases[1]
        a2 = self.sigmoid(z2)
        return np.argmax(a2)

    def sigmoid(self, z):
        # Sigmoid activation function
        z = np.clip(z, -500, 500)  # Clip values to avoid overflow
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        # Derivative of the sigmoid function
        return a * (1 - a)


def load_data(train, test):
    # Load and prepare the training data
    train_set = np.loadtxt(train, skiprows=1, delimiter=',')
    train_data = train_set[:, 1:]
    train_labels = train_set[:, 0]
    training_data = list(zip(train_data, train_labels))

    # Load and prepare the testing data
    test_set = np.loadtxt(test, skiprows=1, delimiter=',')
    test_data = test_set[:, 1:]
    test_labels = test_set[:, 0]
    testing_data = list(zip(test_data, test_labels))

    return training_data, testing_data

def main():
    # Run program: python3 nn.py 784 30 10 fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz
    # Set hyperparameters
    epochs = 30
    learning_rate = 7
    batch_size = 40
    testing = True

    if testing:
        NInput = 784
        NHidden = 30
        NOutput = 10
        train_file = "fashion-mnist_train.csv.gz"
        test_file = "fashion-mnist_test.csv.gz"
    else:
        NInput = sys.argv[1]
        NHidden = sys.argv[2]
        NOutput = sys.argv[3]
        train_file = sys.argv[4]
        test_file = sys.argv[5]
    
    print("Loading training and testing data...")
    training_data, testing_data = load_data(train_file, test_file)
    print("Data Loaded...")

    print("Training the Neural Network...")
    
    # Create and train the neural network
    nn = NeuralNetwork()
    learning_rates = [0.001, 0.01, 1.0, 10, 100]
    minibatch_sizes = [1, 5, 20, 100, 300]

    print("<HYPERPARAMETERS> | Learning Rate = ", learning_rate, "| Batch Size = ", batch_size, "| Cost Function: Quadratic")
    nn.init_parameters(NInput, NHidden, NOutput, epochs, learning_rate, batch_size)
    accuracy = nn.train(training_data, testing_data)
    plt.plot(range(1, epochs+1), accuracy, '-o', label=f"LR = {learning_rate}")
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/100:.0f}%'.format(y)))
    plt.title('Experiment 4: Achieveing Highest Accuracy')
    plt.show()

main()

    #####################################
    # Code for Task 2 and Task 3 Graphs #
    #####################################

    # for i in learning_rates:
    #     print("Task 2")
    #     print("<HYPERPARAMETERS> | Learning Rate = ", i, "| Batch Size = ", batch_size, "| Cost Function: Cross Entropy")
    #     nn.init_parameters(NInput, NHidden, NOutput, epochs, i, batch_size)
    #     accuracies = nn.train(training_data, testing_data)
    #     plt.plot(range(1, epochs+1), accuracies, '-o',  label=f"LR = {i}")
    # plt.grid()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/100:.0f}%'.format(y)))
    # plt.title('Test Accuracy vs Epoch for Different Learning Rates')
    # plt.legend()
    # plt.show()
    
    # acc = []
    # for i in minibatch_sizes:
    #     print ("Task 3")
    #     print("<HYPERPARAMETERS> | Learning Rate = ", learning_rate, "| Batch Size = ", i, "| Cost Function: Cross Entropy")
    #     nn.init_parameters(NInput, NHidden, NOutput, epochs, learning_rate, i)
    #     accuracies = nn.train(training_data, testing_data)
    #     acc.append(max(accuracies))

    # plt.plot(minibatch_sizes, acc, '-o')
    # plt.grid()
    # plt.xlabel('Batch Size')
    # plt.ylabel('Accuracy')
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/100:.0f}%'.format(y)))
    # plt.title('Test Accuracy vs Batch Size for Different Learning Rates')
    # plt.show()
