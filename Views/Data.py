import random
import Global_Variables as global_vars

class Data_Set:
    def __init__(self, 
                x_min = 0,
                x_max = global_vars.X_MAX,
                y_min = 0,
                y_max = global_vars.Y_MAX) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min= y_min
        self.y_max = y_max

    def generate_delimiter(self):
        self.intercept = random.uniform(self.y_min, self.y_max)

        x_range = self.x_max - self.x_min
        min_slope = (self.y_min - self.intercept) / x_range
        max_slope = (self.y_max - self.intercept) / x_range
        # Creates a slope that will not make the line go outside of the graph
        self.slope = (random.uniform(min_slope, max_slope))

    def separation_function(self, x):
        return self.slope * x + self.intercept
        # return -0.3608011240889337 * x + 58.523196966583725
        # return (((x-50)**2)/-30) + 90 # formula for testing non-linear functions
        # return x**2
        # return 10*(x**.5)
        # return .01*(x**2)

    def desired(self, x, y):
        if y > self.separation_function(x):
            return global_vars.X_MAX
        else:
            return 0

    def generate_training_set(self, size):
        x_points = []
        y_points = []
        for i in range(size):
            x_points.append(random.uniform(self.x_min, self.x_max))
            y_points.append(random.uniform(self.y_min, self.y_max))
        return [x_points, y_points]

    def generate_testing_set(self, size):
        x_test_cases = []
        y_test_cases = []
        for i in range(size):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            x_test_cases.append(x)
            y_test_cases.append(y)
        return [x_test_cases, y_test_cases]
