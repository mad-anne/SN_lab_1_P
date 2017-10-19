from abc import ABC, abstractmethod
import random
from math import sqrt

from dataset import split_data_set


class BinaryClassifier(ABC):
    def __init__(self, alpha, bias, act_func):
        self.alpha = alpha
        self.bias = bias
        self.weights = []
        self.w0 = 0.00
        self.act_func = act_func

    def learn(self, epochs, data_set):
        for epoch in range(epochs):
            weights_before = self.weights
            self.learn_epoch(data_set)
            if weights_before == self.weights or self.has_final_condition():
                return epoch + 1
        else:
            return epochs

    def validate(self, data_set):
        return sum([self.predict(d) == d.label for d in data_set]) / len(data_set)

    def get_net(self, data):
        return sum([x * w for x, w in zip(data, self.weights)]) + self.w0 * self.bias

    def predict(self, data):
        return self.act_func.get_output(self.get_net(data.data))

    def init_random_weights(self, size, deviation):
        self.weights = [random.uniform(-deviation, deviation) for i in range(size)]
        self.w0 = random.uniform(-deviation, deviation)

    @abstractmethod
    def learn_epoch(self, data):
        pass

    @abstractmethod
    def get_error(self, output, label):
        pass

    @abstractmethod
    def update_weights(self, data, error):
        pass

    @abstractmethod
    def has_final_condition(self):
        pass


class Perceptron(BinaryClassifier):
    def learn_epoch(self, data_set):
        for data in data_set:
            net = self.get_net(data.data)
            output = self.act_func.get_output(net)
            error = self.get_error(output=output, label=data.label)
            self.update_weights(data.data, error)

    def get_error(self, output, label):
        return label - output

    def update_weights(self, data, error):
        coefficient = self.alpha * error
        self.weights = [w + coefficient * d for w, d in zip(self.weights, data)]
        self.w0 += coefficient * self.bias

    def has_final_condition(self):
        return False


class Adaline(BinaryClassifier):
    def __init__(self, alpha, bias, act_func, min_mse):
        super(Adaline, self).__init__(alpha, bias, act_func)
        self.min_mse = min_mse
        self.curr_mse = min_mse + 1.0

    def learn_epoch(self, data_set):
        errors_sum = 0
        self.curr_mse = 0
        for data in data_set:
            net = self.get_net(data.data)
            error = self.get_error(output=net, label=data.label)
            errors_sum += error * error
            self.update_weights(data.data, error)
        self.curr_mse = errors_sum / len(data_set)

    def get_error(self, output, label):
        return label - output

    def update_weights(self, data, error):
        coefficient = 2 * self.alpha * error
        self.weights = [w + coefficient * d for w, d in zip(self.weights, data)]
        self.w0 += coefficient * self.bias

    def has_final_condition(self):
        return self.curr_mse <= self.min_mse


def cross_validation(classifier, validations, epochs, data_set, data_size, deviation, train_part, validate_part, test_part):
    epochs_sum = 0
    accuracy_sum = 0
    deviations = []

    for v in range(validations):
        random.shuffle(data_set)
        train_set, validate_set, test_set = split_data_set(data_set, train_part, validate_part, test_part)
        classifier.init_random_weights(data_size, deviation)
        epochs_sum += classifier.learn(epochs=epochs, data_set=train_set)
        deviations.append(classifier.validate(validate_set))
        accuracy_sum += deviations[-1]

    result = {}
    accuracy = accuracy_sum / validations
    result['accuracy'] = accuracy
    result['epochs'] = epochs / validations
    result['deviation'] = sqrt(sum([pow(d - accuracy, 2) for d in deviations]) / validations)
    return result
