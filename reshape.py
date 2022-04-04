import numpy as np


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.i_s = input_shape
        self.o_s = output_shape

    def forward(self, input):
        return np.reshape(input, self.o_s)

    def backward(self, output):
        return np.reshape(output, self.i_s)



