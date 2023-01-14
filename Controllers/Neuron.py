from Controllers.Parameter import Parameter
import Global_Variables as global_vars

class Neuron:
    def __init__(self, parent, num_inputs) -> None:
        self.index = 0
        self.Layer = parent
        self.num_inputs = num_inputs
        self.parameters = []
        self.downstream_neurons = []
        for _ in range(num_inputs):
            self.create_Parameter(False)
        self.create_Parameter(True)

    # def activate(self, inputs):
    #     sum = 0
    #     for i in range(len(inputs)):
    #         sum += inputs[i] * self.weights[i]
    #     sum += self.bias
    #     return sum

    def to_string(self):
        s = "Number of inputs: " + str(self.num_inputs)
        return s

    # TODO these might all be redundant functions since they only have 1 line each
    def create_Parameter(self, bias):
        self.parameters.append(Parameter(self, bias))

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.parameters):
            current = self.parameters[self.index]
            self.index += 1
            return current
        self.index = 0
        raise StopIteration

    def activate(self, inputs):
        inputs.append(1)
        sum = 0.0
        for input, parameter in zip(inputs, self.parameters):
            parameter.input = input
            sum += input * parameter.value
        return self._normalize_output(sum)

    def _normalize_output(self, sum):
        if global_vars.ACTIVATION_FUNCTION == "Open":
            return sum