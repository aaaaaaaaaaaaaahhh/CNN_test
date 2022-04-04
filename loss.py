import numpy as np


class loss:
    def __init__(self, y):
        self.y = y

    def cross_entropy(self, y_pred):
        self.y_pred = y_pred
        print(self.y_pred[0])
        print(np.log10(self.y_pred[0]))
        return -np.sum(self.y * np.log10(self.y_pred))

    def cross_entropy_prime(self, y_pred):
        self.y_pred = y_pred
        dLdY = self.y_pred - self.y
        return dLdY


