"""

Author: Tranter Tech
Date: 2024
"""
import numpy as np


class BaseInterface:
    def __init__(self):
        pass

    def run_initialisation(self, args=None):
        return True

    def run_parameters(self, params, args=None):
        pass


class TestInterface(BaseInterface):
    def __init__(self):
        super(TestInterface, self).__init__()

    def run_parameters(self, params, args=''):
        return self._cost_function(params, args)

    def _cost_function(self, params, args):
        params = np.array(params)
        if args == 'ackley':
            val = self.ackley_func(params)
        else:
            raise RuntimeError(f'Parameter for function {args} does not match any function.')
        return val

    @staticmethod
    def ackley_func(chromosome):
        """"""
        firstSum = 0.0
        secondSum = 0.0
        for c in chromosome:
            firstSum += c ** 2.0
            secondSum += np.cos(2.0 * np.pi * c)
        n = float(len(chromosome))
        return -20.0 * np.exp(-0.2 * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + 20 + np.exp(1)

