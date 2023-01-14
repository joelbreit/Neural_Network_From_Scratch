from Controllers.Layer import Layer
import Global_Variables as global_vars
from Views.Utilities import Logger

class Network:
    def __init__(self, num_hidden_layers, neurons_per_layer):
        self.index = 0
        self.layers = []
        self.inputs = [0,0]
        self.neurons_per_layer = neurons_per_layer
        self.num_hidden_layers =num_hidden_layers
        self.num_max_parameters = max(neurons_per_layer, 2) + 1 # TODO no magic numbers
        if num_hidden_layers > 0:
            # Create first hidden layer
            self.create_Layer(2, neurons_per_layer) # Assuming 2 inputs
            # Create remaining hidden layers
            for _ in range(num_hidden_layers - 1):
                self.create_Layer(neurons_per_layer, neurons_per_layer)
        # Create output layer
        self.create_Layer(neurons_per_layer, 1) # Assuming 1 output

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.layers):
            current = self.layers[self.index]
            self.index += 1
            return current
        self.index = 0
        raise StopIteration

    def create_Layer(self, num_inputs, num_neurons):
        self.layers.append(Layer(self, num_inputs, num_neurons))
        # if len(self.layers) > 1:
        #     for upstream_neuron in self.layers[-2]:
        #         for downstream_neuron in self.layers[-1]:
        #             upstream_neuron.downstream_neurons = downstream_neuron

    def activate(self, inputs):
        self.inputs = inputs
        for layer in self.layers:
            outputs = layer.activate(inputs)
            inputs = outputs # The last layer's outputs are the next layers inputs
        return outputs[0]

    def train(self, inputs, desired_output):
        self.inputs = inputs
        # Activate network
        guess = self.activate(inputs)
        # Calculate total loss
        desired_network_adjsutment = desired_output - guess
        # Calculate partial leverage values
        # Calculate aggregate leverages and update parameter values

        leverage_multiplier = len(self.layers)
        # for each parameter, calulate it's total leverage
        for layer in self:
            for neuron in layer:
                for parameter in neuron:
                    parameter.temp = parameter.value # TODO probably don't need to store this in the parameter object
                    parameter.value += global_vars.LEARNING_RATE * leverage_multiplier * 100
                    new_guess = self.activate(inputs)
                    parameter.aggregate_leverage = new_guess - guess
                    # Logger.info("Layer", self.index, "Neuron", layer.index, "Parameter", neuron.index, "Guess", guess, "New Guess", new_guess, "Leverage:", parameter.aggregate_leverage)
                    parameter.value = parameter.temp
            leverage_multiplier -= 1
        for layer in self:
            for neuron in layer:
                for parameter in neuron:
                    parameter.value += global_vars.LEARNING_RATE * abs(parameter.input) * desired_network_adjsutment * parameter.aggregate_leverage

        return self.activate(inputs) - guess

    def guess(self, inputs):
        output = self.activate(inputs)
        if output >= (global_vars.X_MAX/2):
            return global_vars.X_MAX
        else:
            return 0

    def display(self):
        precision = 5
        height = self.neurons_per_layer * (self.num_max_parameters + 2)
        width = (30 * self.num_hidden_layers) + 20

        neuron_column = [""] * height


        vertical_positions = {
            "x": round(height * (1/3)),
            "y": round(height * (2/3)),
            # "output":
        }
        for layer in self:
            for neuron in layer:
                position = (layer.index-1) * (self.num_max_parameters + 2)
                neuron_column[position] += "{:30}".format("Neuron " + str(self.index) + " " + str(layer.index))
                # position += 1
                for i in range(len(neuron.parameters)):
                    weight_column = "W" + str(i) + ": " + str(round(neuron.parameters[i].value, precision))
                    leverage_column = "L" + str(i) + ": " + str(round(neuron.parameters[i].aggregate_leverage, precision))
                    row = "{:13}".format(weight_column)
                    row += "{:13}".format(leverage_column)
                    neuron_column[position+i+1] += "{:30}".format(row)
                # neuron_column[position+2] += "{:30}".format("W2: " + str(round(neuron.parameters[1].value, precision)) + " L2: " + str(round(neuron.parameters[1].aggregate_leverage, precision)))
                # neuron_column[position+3] += "{:30}".format("Bs: " + str(round(neuron.parameters[2].value, precision)) + " BL: " + str(round(neuron.parameters[2].aggregate_leverage, precision)))
                

        display = "Network Visualization:\n"
        input_column = []
        for row_index in range(height):
            row = ""
            # Input column
            if row_index == vertical_positions["x"]:
                row += "{:10}".format("x: " + str(round(self.inputs[0], 3)))
            elif row_index == vertical_positions["y"]:
                row += "{:10}".format("y: " + str(round(self.inputs[1], 3)))
            else:
                row += "{:10}".format("")
            input_column.append(row)

        # Combine columns
        for i in range(height):
            row = "{:10}".format(input_column[i])
            row += "{:10}".format(neuron_column[i])
            # for n in neuron_column:
            #     row += "{:10}".format(n[i])
            # row += "{:30}".format(output_column[i])
            display += row + "\n"


        Logger.info(display)
