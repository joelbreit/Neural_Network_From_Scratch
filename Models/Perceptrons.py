threshold = 1.5
inputs = [1, 0, 1, 0, 1]
weights = [0.7, 0.6, 0.5, 0.3, 0.4]

sum = 0
for i in range(0,len(inputs)):
  sum += inputs[i] * weights[i]

activate = (sum > 1.5)

print(activate)
