import random
import Global_Variables as global_vars


class Parameter:
    def __init__(self, parent, bias=False) -> None:
        self.Neuron = parent
        if bias:
            self.value = random.uniform(-10,10)
        else:
            self.value = random.uniform(-1, 1) #TODO no magic numbers
        self.temp = None
        self.partial_leverage = 0
        self.aggregate_leverage = 0
        self.input = None

    # def calculate_partial_leverage(self):
    #     if global_vars.ACTIVATION_FUNCTION == "Open":
    #         # f(x) = a*x + global_varsants where a is the last input and x is the value weight of this parameter
    #         self.partial_leverage = input

    # def calculate_aggregate_leverage(self):
    #     if len(self.downstream_neurons) == 0:
    #         return self.partial_leverage
    #     else:
    #         # chain rule
    #         sum = 0.0
    #         for downstream_neuron in self.Neuron.downsteam_neurons:
    #             sum += self.partial_leverage * downstream_neuron.
    #         return self.partial_leverage + 