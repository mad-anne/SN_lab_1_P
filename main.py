import json

from activation_function import UnipolarStepFunction
from dataset import Data
from classifier import Perceptron

with open('parameters.json', 'r') as f:
    params = json.loads(f.read())

alpha = params["alpha"]
bias = params["bias"]
data_size = params["dataSize"]
epochs = params["epochs"]
weights_deviation = params["randomWeightsDeviation"]

perceptron = Perceptron(alpha=alpha, bias=bias)

data_set = [
    Data([0, 0], 0),
    Data([0, 1], 0),
    Data([1, 0], 0),
    Data([0, 0], 0),
    Data([0, 1], 0),
    Data([1, 0], 0),
    Data([1, 1], 1),
    Data([1, 1], 1),
]

act_func = UnipolarStepFunction()

perceptron.init_random_weights(data_size, -weights_deviation, weights_deviation)
perceptron.learn(epochs=epochs, data_set=data_set, act_func=act_func)

print(perceptron.validate(data_set, act_func))
