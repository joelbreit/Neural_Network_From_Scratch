from random import random
import numpy as np

import Global_Variables as global_vars
from Views.Utilities import Logger


class Perceptron:
    def __init__(self, num_inputs = 2, num_outputs = 1) -> None:
        self.num_inputs = num_inputs
        self.num_ouputs = num_outputs
        self.weights = []
        for _ in range(num_inputs):
            self.weights.append(random() * 2 - 1) # random weigth between -1 and 1
        self.weights.append(random() * 2 - 1) # extra weight is for the bias
        self.bias = 1

        # global_vars.LEARNING_RATE = 0.001

    def activate(self, inputs):
        inputs.append(self.bias)
        sum = 0
        for input, weight in zip(inputs, self.weights):
            sum += input * weight
        return sum

    def guess(self, inputs):
        sum = self.activate(inputs)
        if sum > global_vars.X_MAX / 2:
            return global_vars.X_MAX
        else:
            return 0

    def train(self, inputs, label):
        needed_adjustment = label - self.activate(inputs)
        # inputs.append(self.bias)
        previous_weights = self.weights
        adjustments = []
        # for input, weight in zip(inputs, self.weights):
        #     adjustment = input * needed_adjustment * global_vars.LEARNING_RATE
        #     adjustments.append(adjustment)
        #     weight += adjustment
        for i in range(len(inputs)):
            adjustment = inputs[i] * needed_adjustment * global_vars.LEARNING_RATE
            adjustments.append(adjustment)
            self.weights[i] += adjustment

        # Logger.info(f"Needed adjustment: {needed_adjustment}")
        # Logger.info(f"Adjusted X Weight from {round(previous_weights[0] ,5)} to {round(self.weights[0] ,5)} (adjustment={adjustments[0]}) {round(inputs[0])} * {round(needed_adjustment, 2)} * {global_vars.LEARNING_RATE}")        
        # Logger.info(f"Adjusted Y Weight from {round(previous_weights[1] ,5)} to {round(self.weights[1] ,5)} (adjustment={adjustments[1]}) {round(inputs[1])} * {round(needed_adjustment, 2)} * {global_vars.LEARNING_RATE}")        
        # Logger.info(f"Adjusted Bias from {round(previous_weights[2] ,5)} to {round(self.weights[2] ,5)} (adjustment={adjustments[2]}) {round(inputs[2])} * {round(needed_adjustment, 2)} * {global_vars.LEARNING_RATE}")        

    def display(self):
        Logger.info(f"X Weight: {self.weights[0]}, Y Weight: {self.weights[1]}, Bias: {self.weights[2]}")