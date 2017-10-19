from classifier import cross_validation
import matplotlib.pyplot as plt


def present(classifier, validations, epochs, data_set, data_size, weights_deviation, train_part, validate_part,
    test_part):
    results = cross_validation(
        classifier, validations, epochs, data_set, data_size, weights_deviation, train_part, validate_part,
        test_part)

    print('Accuracy: %.2f %% (+/- %.2f)' % (results['accuracy'] * 100, results['deviation']))
    print('Epochs average: %.2f' % results['epochs'])


# plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()
