from classifier import cross_validation
from dataset import Data
import matplotlib.pyplot as plt


def present_variants(
        data_sets, classifiers, validations, epochs, data_size, weights_deviation, train_part, validate_part,
        test_part, accuracies, take_user_inputs):
    for logic_func, data_set in data_sets.items():
        for label, classifier in classifiers.items():
            print("\nLearning %s with %s" % (logic_func, label))
            present(
                classifier, label, logic_func, validations, epochs, data_set, data_size, weights_deviation, train_part,
                validate_part, test_part, accuracies)
            if take_user_inputs:
                process_user_inputs(classifier)


def present(
        classifier, label, logic_func, validations, epochs, data_set, data_size, weights_deviation, train_part,
        validate_part, test_part, accuracies):
    results = cross_validation(
        classifier, validations, epochs, data_set, data_size, weights_deviation, train_part, validate_part,
        test_part)

    print('Accuracy: %.2f %% (+/- %.2f)' % (results['accuracy'] * 100, results['deviation']))
    print('Epochs average: %.2f' % results['epochs'])

    key = label + " " + logic_func
    if key not in accuracies:
        accuracies[key] = []
    accuracies[key].append(results['accuracy'])


def show_accuracies_plot(accuracies, x_labels):
    x_range = [i for i in range(len(x_labels))]
    for key, value in accuracies.items():
        plt.plot(x_range, value, label=key)
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(x_range, x_labels)
    axes = plt.gca()
    axes.set_ylim([0.0, 1.1])
    plt.show()


def process_user_inputs(classifier):
    while True:
        print("\nGive your inputs to validate or press ENTER to continue")
        values = classifier.get_data_size()
        data = []
        for i in range(values):
            text = 'Enter value %s: ' % (i + 1)
            _data = input(text)
            if not _data:
                return
            data.append(float(_data))
        predicted_value = classifier.predict(Data(data, None))
        print('Predicted value: %s' % predicted_value)

