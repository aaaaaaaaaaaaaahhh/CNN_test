import numpy as np

X = np.array([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]])

class max_pool_layer:

  def __init__(self, input_size, f_size):
    self.d, self.m, self.n = input_size # defining the depth and number of rows and columns of the input. The depth of the filters is d as well
    self.f_m, self.f_n= f_size # defining number of rows and columns in a filter
    self.out_m = int((self.m - self.f_m)/self.f_m + 1) # using formula to get the size of the output
    self.out_n = int((self.n - self.f_n)/self.f_n + 1)
    self.out = np.zeros((self.d, self.out_m, self.out_n))
    self.input = None

  def forward(self, input):
    self.input = input
    #print("input shape to max pooling", np.shape(input))
    for j in range(self.d): # for each layer of the input
        for row in range(self.out_m): # row and column are regular 2d loop
          for col in range(self.out_n):
            self.out[j, row, col] = np.amax(self.input[j, row*self.f_m:row*self.f_m+self.f_m, col*self.f_n:col*self.f_n+self.f_n])
            #print(self.input[j, row:row+self.f_m, col:col+self.f_n])
    return self.out

  def backward(self):
    self.input_gradient = np.zeros((self.d, self.m, self.n))
    for j in range(self.d): # for each layer of the input
      for row in range(self.out_m): # row and column are regular 2d loop
        for col in range(self.out_n):
          max_index = np.where(self.input==np.amax(self.input[j, row*self.f_m:row*self.f_m+self.f_m, col*self.f_n:col*self.f_n+self.f_n]))
          self.input_gradient[max_index[0], max_index[1], max_index[2]] = 1 #setting indexes in input gradient to the derivatives of input
          #print(self.input[j, row:row+self.f_m, col:col+self.f_n])
    return self.input_gradient




