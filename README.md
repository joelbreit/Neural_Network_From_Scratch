# Neural_Network
 
Requires matplotlib and numpy

## Running the code

Currently, this is a bit sketchy. Can run: `python __main__.py` You might need to change that to `python3 __main__.py` if not using an Anaconda setup.

The perceptron module doesn't work as set up.

## Issues

* Currently only reaches 80-90%ish accuracy at 5000 training points and doesn't seem to really improve from there.
* Should probably save charts as photos instead of displaying them at the end because the chart has to be closed everytime the code successfully runs
* Occassionally, the weights get too big and crash the program or too small and zero out causing deadlock in the training process
* The line does not plot when an X_MAX of 1 is used
* Currently multiplying the output by 100 to get around some weird bugs
* The neural network online creates straight lines which seems to imply that it is effectively only simulating 1 neuron?
* The program takes quite a while to reach 80%ish accuracy, which implies that something is mushing up the results

## Things to add

* Batch sizes
* Make runnable as a module
* Better logging
    * Log rolling accuracy numbers
    * Create a log file for each run
    * Create a log function to visualize a neural network