from activation_function import BipolarStepFunction, UnipolarStepFunction
from dataset import read_data_set, enlarge_data_set
from classifier import Perceptron, Adaline
from parameters import read_parameters
from presenter import present


params = read_parameters('parameters.json')

alpha = params["alpha"]
bias = params["bias"]
min_mse = params["minMSE"]
data_size = params["dataSize"]
validations = params["validations"]
epochs = params["epochs"]
weights_deviation = params["randomWeightsDeviation"]
enlarge_times = params["enlargeDataSetTimes"]
enlarge_deviation = params["enlargedDataSetMaxDeviation"]
train_part = params["trainingSetPart"]
validate_part = params["validatingSetPart"]
test_part = params["testingSetPart"]

print(
    "\nParameters: alpha = %.2f, epochs = %d, weights deviation = %.2f"
    % (alpha, epochs, weights_deviation))

classifiers_unipolar = {
    "Unipolar Perceptron": Perceptron(alpha=alpha, bias=bias, act_func=UnipolarStepFunction()),
    "Unipolar Adaline": Adaline(alpha=alpha, bias=bias, min_mse=min_mse, act_func=UnipolarStepFunction())
}

classifiers_bipolar = {
    "Bipolar Perceptron": Perceptron(alpha=alpha, bias=bias, act_func=BipolarStepFunction()),
    "Bipolar Adaline": Adaline(alpha=alpha, bias=bias, min_mse=min_mse, act_func=BipolarStepFunction())
}

data_sets_unipolar = {
    "AND": enlarge_data_set(read_data_set('data_set_and_unipolar.txt'), enlarge_times, enlarge_deviation),
    "OR": enlarge_data_set(read_data_set('data_set_or_unipolar.txt'), enlarge_times, enlarge_deviation)
}

data_sets_bipolar = {
    "AND": enlarge_data_set(read_data_set('data_set_and_bipolar.txt'), enlarge_times, enlarge_deviation),
    "OR": enlarge_data_set(read_data_set('data_set_or_bipolar.txt'), enlarge_times, enlarge_deviation)
}


for logic_func, data_set in data_sets_unipolar.items():
    for label, classifier in classifiers_unipolar.items():
        print("\nLearning %s with %s" % (logic_func, label))
        present(
            classifier, validations, epochs, data_set, data_size, weights_deviation, train_part, validate_part,
            test_part)


for logic_func, data_set in data_sets_bipolar.items():
    for label, classifier in classifiers_bipolar.items():
        print("\nLearning %s with %s" % (logic_func, label))
        present(
            classifier, validations, epochs, data_set, data_size, weights_deviation, train_part, validate_part,
            test_part)
