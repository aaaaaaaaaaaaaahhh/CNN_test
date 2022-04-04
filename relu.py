import numpy as np

class ReLU:

  def forward(self, z):
    self.z = z
    return np.maximum(-0.1*self.z, self.z)

  def backward(self, dLdA):
    self.dLdA = dLdA
    return np.where(dLdA >= 0, 1, -0.1*dLdA)




