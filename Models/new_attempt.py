import random
import math

class Neuron:
    def __init__(self, num_inputs) -> None:
        for _ in range(num_inputs):
            self.weights.append(random.uniform(-1, 1))
        self.bias = random.uniform(-1, 1)

    def activate(self, inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        sum += self.bias
        return (1 / (1 + math.exp(-sum)))*2 - 1

class Network:
    def __init__(self, num_inputs, num_hidden_layers, neurons_per_layer) -> None:
        self.layers = []        
        # Create first hidden layer
        self.layers.append(2, neurons_per_layer) # Assuming 2 layers
        # Create remaining hidden layers
        for _ in range(num_hidden_layers - 1):
            # self.layers.append(self.Layer(self))
            self.create_Layer(neurons_per_layer, neurons_per_layer)
        # Create output layer
        self.create_Layer(neurons_per_layer, 1) # Assuming 1 output


    def activate():
        for layer in layers

    def train():
        pass

    def show():
        pass