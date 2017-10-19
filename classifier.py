from abc import ABC, abstractmethod
import random


class BinaryClassifier(ABC):
    def __init__(self, alpha, bias):
        self.alpha = alpha
        self.bias = bias
        self.weights = []
        self.w0 = 0.00

    def learn(self, epochs, data_set, act_func):
        for epoch in range(epochs):
            weights_before = self.weights
            self.learn_epoch(data_set, act_func)
            if weights_before == self.weights or self.has_final_condition():
                print('Learned in %s epochs' % (epoch + 1))
                break
        else:
            print('Learned in %s epochs' % epochs)

    def get_net(self, data):
        return sum([x * w for x, w in zip(data, self.weights)]) + self.w0 * self.bias

    def predict(self, data, act_func):
        return act_func.get_output(self.get_net(data.data))

    def init_random_weights(self, size, min_val, max_val):
        self.weights = [random.uniform(min_val, max_val) for i in range(size)]
        self.w0 = random.uniform(min_val, max_val)

    @abstractmethod
    def learn_epoch(self, data, act_func):
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
    def learn_epoch(self, data_set, act_func):
        for data in data_set:
            net = self.get_net(data.data)
            output = act_func.get_output(net)
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
    def __init__(self, alpha, bias, min_mse):
        super(Adaline, self).__init__(alpha, bias)
        self.min_mse = min_mse
        self.curr_mse = min_mse + 1.0

    def learn_epoch(self, data_set, act_func):
        errors_sum = 0
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
