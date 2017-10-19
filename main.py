from activation_function import UnipolarStepFunction
from dataset import read_data_set, enlarge_data_set
from classifier import Perceptron, cross_validation
from parameters import read_parameters

params = read_parameters('parameters.json')

alpha = params["alpha"]
bias = params["bias"]
data_size = params["dataSize"]
validations = params["validations"]
epochs = params["epochs"]
weights_deviation = params["randomWeightsDeviation"]
enlarge_times = params["enlargeDataSetTimes"]
enlarge_deviation = params["enlargedDataSetMaxDeviation"]
train_part = params["trainingSetPart"]
validate_part = params["validatingSetPart"]
test_part = params["testingSetPart"]

data_set_and_unipolar = read_data_set('data_set_and_unipolar.txt')
enlarge_data_set(data_set_and_unipolar, enlarge_times, enlarge_deviation)

perceptron = Perceptron(alpha=alpha, bias=bias)

act_func = UnipolarStepFunction()

result = cross_validation(
    perceptron, validations, epochs, data_set_and_unipolar, act_func, data_size, weights_deviation, train_part,
    validate_part, test_part)


def present_results(results):
    print('Accuracy: %.2f %% (+/- %.2f)' % (results['accuracy'] * 100, results['deviation']))
    print('Epochs average: %.2f' % results['epochs'])

present_results(result)