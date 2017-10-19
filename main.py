from activation_function import UnipolarStepFunction
from dataset import Data
from classifier import Perceptron

perceptron = Perceptron(alpha=0.1, bias=1.0)

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

epochs = 10
perceptron.init_random_weights(2, -0.2, 0.2)
perceptron.learn(epochs=epochs, data_set=data_set, act_func=act_func)

for data in data_set:
    print(perceptron.predict(data, act_func))
