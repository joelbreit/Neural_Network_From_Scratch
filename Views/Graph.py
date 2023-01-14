import matplotlib.pyplot as plt # for graphically displaying results
import numpy as np # for more easily graphing a line
import random # for generating a test set
import Global_Variables as global_vars

class Graph:
    def __init__(self, 
                num_training = 5000, 
                num_testing = 10000, 
                x_max = global_vars.X_MAX,
                y_max = global_vars.Y_MAX,
                x_min = 0,
                y_min = 0) -> None:
        self.num_traing = num_training
        self.num_testing = num_testing
        self.x_max = x_max
        self.y_max = y_max
        self.x_min = x_min
        self.y_min= y_min

    def create_reference_line(self, x, y):
        plt.plot(x, y, color='black')

    def create_training_set(self):
        self.x_points = []
        self.y_points = []
        for i in range(self.num_training):
            self.x_points.append(random.random() * self.x_max)
            self.y_points.append(random.random() * self.y_max)

    def show(self, x_positives, y_positives, x_negatives, y_negatives):

        x = np.array(x_positives)
        y = np.array(y_positives)
        plt.scatter(x, y, color='green', s=1)

        x = np.array(x_negatives)
        y = np.array(y_negatives)
        plt.scatter(x, y, color='red', s=1)
        plt.show()

    def show(self, predictions):
        xs = []
        ys = []
        colors = []
        for p in predictions:
            xs.append(p[0])
            ys.append(p[1])
            normalize = max(min(p[2]/global_vars.X_MAX, 1), 0)
            red = 1 - normalize
            green = normalize
            colors.append([(red, green, 0, 0.5)])
            # if p[2] > 0.5:
            #     colors.append('green')
            # else:
            #     colors.append('red')
        x = np.array(xs)
        y = np.array(ys)
        c = np.array(colors)
        # plt.scatter(p[0], p[1], color=[(red, 0, green)], s=1)
        plt.scatter(x, y, c=c, s=1)

        plt.show()