from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def get_output(self, func_input):
        pass


class UnipolarStepFunction(ActivationFunction):
    def get_output(self, func_input):
        return 0.0 if func_input < 0 else 1.0


class BipolarStepFunction(ActivationFunction):
    def get_output(self, func_input):
        return -1.0 if func_input < 0 else 1.0
