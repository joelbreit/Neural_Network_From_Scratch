import sys # for taking arguments from the command line
from Views.Utilities import Logger
from Controllers.Network import Network
from Views.Graph import Graph
from Views.Data import Data_Set
import numpy as np
import Global_Variables as global_vars

# TODO set this tester up with actual AI libraries to see how much better they do
# TODO Create unit tests

# Set up logging
# TODO add more command line arguments
command_line_arguments = sys.argv
debug = True
if "-v" in command_line_arguments or "--verbose" in command_line_arguments:
    debug = True

# Global variable
if "-l" in command_line_arguments:
    global_vars.LEARNING_RATE = command_line_arguments[command_line_arguments.index("-l") + 1]
elif "--learning_rate" in command_line_arguments:
    global_vars.LEARNING_RATE = command_line_arguments[command_line_arguments.index("--learning_rate") + 1]

Logger.debug("learning_rate =", global_vars.LEARNING_RATE)
Logger.debug("Arguments:", command_line_arguments)

x_min = 0
y_min = 0

data = Data_Set(x_min, global_vars.X_MAX, y_min, global_vars.Y_MAX)
data.generate_delimiter()
graph = Graph(global_vars.NUM_TRAINING, global_vars.NUM_TESTING, global_vars.X_MAX, global_vars.Y_MAX, x_min, y_min)

Logger.debug("Slope:", data.slope)
Logger.debug("Intercept:", data.intercept)

x = np.linspace(x_min, global_vars.X_MAX, global_vars.X_MAX)
y = data.separation_function(x)
graph.create_reference_line(x, y)

training_set = data.generate_training_set(global_vars.NUM_TRAINING)
x_points = training_set[0]
y_points = training_set[1]

network = Network(num_hidden_layers=3, neurons_per_layer=5) # 1 x value and 1 y value always means 2 inputs

network.display()
Logger.info("hello")

# TODO move this to another file
def test(chart = False):
    # Assess results
    training_data = data.generate_testing_set(global_vars.NUM_TESTING)
    num_right = 0
    for x, y in zip(training_data[0], training_data[1]):
        if data.desired(x, y) == network.guess([x, y]):
            num_right += 1
    percent_correct = round((num_right / global_vars.NUM_TESTING) * 100, 2)
    network.display()
    Logger.info("Neuron is ", percent_correct, "% correct", delimiter='')

    x_positives = []
    y_positives = []
    x_negatives = []
    y_negatives = []

    # Display results
    iterator = 0
    for x, y in zip(training_data[0], training_data[1]):
        Logger.display_progress("Generating graph: ", iterator, global_vars.NUM_TESTING)
        guess = network.activate([x, y])
        if guess >= (global_vars.X_MAX/2):
            x_positives.append(x)
            y_positives.append(y)
        else:
            x_negatives.append(x)
            y_negatives.append(y)
        iterator += 1

    Logger.debug(len(x_positives), "positive values")
    Logger.debug(len(x_negatives), "negative values")
    # network.show()

    if chart:
        graph.show(x_positives, y_positives, x_negatives, y_negatives)

# Train data
for j in range(global_vars.NUM_ITERATIONS):
    for i in range(global_vars.NUM_TRAINING):
        Logger.display_progress("Training: ", i, global_vars.NUM_TRAINING)
        network.train([x_points[i], y_points[i]], data.desired(x_points[i], y_points[i]))
        if i % 1000 == 0:
            test()
            global_vars.LEARNING_RATE /= 2 # Simulated annealing
Logger.display_progress("Training: ", global_vars.NUM_ITERATIONS, global_vars.NUM_ITERATIONS, final=True)
test(True)