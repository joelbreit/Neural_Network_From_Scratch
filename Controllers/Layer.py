from Controllers.Neuron import Neuron

class Layer:
    def __init__(self, parent, num_inputs, num_neurons) -> None:
        self.Network = parent
        self.index = 0
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.neurons = []
        for _ in range(num_neurons):
            self.create_Neuron(num_inputs)
            
    def to_string(self):
        s = "Number of inputs: " + str(self.num_inputs) + "\nNumber of neurons: " + str(self.num_neurons)
        return s

    def create_Neuron(self, num_inputs):
        self.neurons.append(Neuron(self, num_inputs))

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.neurons):
            current = self.neurons[self.index]
            self.index += 1
            return current
        self.index = 0
        raise StopIteration

    def activate(self, inputs):
        output = []
        for neuron in self.neurons:
            output.append(neuron.activate(inputs))
        return output