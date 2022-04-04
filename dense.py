import numpy as np

# if you dont get any of the derivation, dont worry i dont really get them either.
# go to https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week7/week7_homework/?activate_block_id=block-v1%3AMITx%2B6.036%2B1T2019%2Btype%40sequential%2Bblock%40week7_homework

np.random.seed(6)

x = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])


class FC_layer:
    def __init__(self, input, nodes):
        self.x = input
        self.d, self.n = np.shape(input)  # d is number of dimensions to the data and n is the number of data points
        # being fed through(batch size)
        self.m = nodes  # number of nodes
        self.weights = np.random.randn(self.m, self.d)  # d by m
        self.biases = np.zeros((self.m, 1))
        self.z = None
        self.dLdA = None  # d by n, A is the inputs to the layer
        self.dLdW = None  # d by n
        self.dLdW0 = None

    def linear(self, forward=True, dLdZ=None, learning_rate=.01):  # dLdZ is of size m(l+1) by n(l+1)
        if forward:
          self.z = (self.weights @ self.x) + self.biases
          return self.z
        else:  # for backprop
          #print("shape of weights, in linear", np.shape(self.weights))
          #print("shape of dLdZ, in linear", np.shape(dLdZ))
          #print("shape of x, in linear", np.shape(self.x))
          #print("shape of biases, in linear", np.shape(self.biases))
          self.dLdA = self.weights.T@dLdZ
          self.dLdW = (dLdZ@self.x.T)
          self.dLdW0 = dLdZ  # m by n (same size as dLdZ)
          #print("dldw", self.dLdW)
          #print("shape of dLdW0, in linear", np.shape(self.dLdW0))
          #print("shape of dLdW, in linear", np.shape(self.dLdW))
          self.weights -= learning_rate*self.dLdW
          self.biases -= learning_rate*self.dLdW0

          return self.dLdA  # d by n

    def reLU(self, z, forward=True, dLdA=None):
        if forward:
          return np.maximum(0, z)
        else:
          return np.where(dLdA >= 0, 1, 0)

    def SoftMax(self, z, forward=True, dLdZ=None):
        if forward:
          out = np.zeros((np.shape(z)))
          self.n, self.k = np.shape(z)
          for j in range(self.k):
            out[:, j] = np.exp(z[:, j])/np.sum(np.exp(z), 0)
          return out
        else:
          return dLdZ

'''
layer_1 = FC_layer(x.T, 5)

Z_1 = layer_1.linear()
print(Z_1)

A_1 = layer_1.reLU(Z_1)
print(A_1)

layer_2 = FC_layer(A_1, 4)

Z_2 = layer_2.linear()
print(Z_2)

A_2 = layer_2.reLU(Z_2)
print(A_2)



Z_1 = layer_1.linear(forward=False)
print(Z_1)

A_1 = layer_1.reLU(forward=False)
print(A_1)


Z_2 = layer_2.linear(forward=False)
print(Z_2)

A_2 = layer_2.reLU(forward=Z_2)
print(A_2)


'''
