from Network import Network

network = Network(num_hidden_layers=2, neurons_per_layer=3)

for layer in network:
    for neuron in layer:
        print(neuron.to_string())

print(network.activate([2, 3]))

change = network.train([5.3, 3.4], desired_output=1)
print("This should be negative:", change)