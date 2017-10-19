from activation_function import BipolarStepFunction, UnipolarStepFunction
from dataset import read_data_set, enlarge_data_set
from classifier import Perceptron, Adaline
from parameters import read_parameters
from presenter import present, show_accuracies_plot, present_variants


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
show_plots = params["showPlots"]
research_alphas = params["researchAlpha"]
research_weights = params["researchWeights"]

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

if research_weights:
    weights_deviations = [x * 0.1 for x in range(0, 11)]

    accuracies = {}

    for weights_deviation in weights_deviations:
        present_variants(
            data_sets_unipolar, classifiers_unipolar, validations, epochs, data_size, weights_deviation, train_part,
            validate_part, test_part, accuracies)

        present_variants(
            data_sets_bipolar, classifiers_bipolar, validations, epochs, data_size, weights_deviation, train_part,
            validate_part, test_part, accuracies)

    if show_plots:
        labels = ['%.2f' % w for w in weights_deviations]
        show_accuracies_plot(accuracies, labels)


if research_alphas:
    alphas = [x * 0.05 for x in range(0, 21)]

    accuracies = {}

    for alpha in alphas:
        classifiers_unipolar = {
            "Unipolar Perceptron": Perceptron(alpha=alpha, bias=bias, act_func=UnipolarStepFunction()),
            "Unipolar Adaline": Adaline(alpha=alpha, bias=bias, min_mse=min_mse, act_func=UnipolarStepFunction())
        }

        classifiers_bipolar = {
            "Bipolar Perceptron": Perceptron(alpha=alpha, bias=bias, act_func=BipolarStepFunction()),
            "Bipolar Adaline": Adaline(alpha=alpha, bias=bias, min_mse=min_mse, act_func=BipolarStepFunction())
        }

        present_variants(
            data_sets_unipolar, classifiers_unipolar, validations, epochs, data_size, weights_deviation, train_part,
            validate_part, test_part, accuracies)

        present_variants(
            data_sets_bipolar, classifiers_bipolar, validations, epochs, data_size, weights_deviation, train_part,
            validate_part, test_part, accuracies)

    if show_plots:
        labels = ['%.2f' % w for w in alphas]
        show_accuracies_plot(accuracies, labels)

accuracies = {}

classifiers_unipolar = {
    "Unipolar Perceptron": Perceptron(alpha=alpha, bias=bias, act_func=UnipolarStepFunction()),
    "Unipolar Adaline": Adaline(alpha=alpha, bias=bias, min_mse=min_mse, act_func=UnipolarStepFunction())
}

classifiers_bipolar = {
    "Bipolar Perceptron": Perceptron(alpha=alpha, bias=bias, act_func=BipolarStepFunction()),
    "Bipolar Adaline": Adaline(alpha=alpha, bias=bias, min_mse=min_mse, act_func=BipolarStepFunction())
}

present_variants(
    data_sets_unipolar, classifiers_unipolar, validations, epochs, data_size, weights_deviation, train_part,
    validate_part, test_part, accuracies)

present_variants(
    data_sets_bipolar, classifiers_bipolar, validations, epochs, data_size, weights_deviation, train_part,
    validate_part, test_part, accuracies)
