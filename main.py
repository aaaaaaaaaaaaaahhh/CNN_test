import numpy as np
import convolutional_layer
import scipy
import max_pooling_layer
import relu
import reshape
import dense
import loss
import copy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# TODO:
# - make it work
# - standardize the layers
#    a. each layer should have a forward and backward function
#    b. variables should be the same name
# - create a train function that loops through the layers and calls backward and forward functions
# - conv2 filter weights are on bigger scale
#    a. layer 3 weights are on different scale than layer 4 weights and conv layer kernels.Why?
# - conv layer weights are repeating.why?
#    a. each f_num layer in the weights is the same, i.e.all the values are the same

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def process(x, y, limit, num_classes):
    x = x[0:limit][:][:]
    y = y[0:limit][:][:]

    x = x.astype("float32") / 255
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    y = to_categorical(y, num_classes)
    return x, y


x_train, y_train = process(x_train, y_train, 19, 10)
x_test, y_test = process(x_test, y_test, 19, 10)

X = np.array([[-1, 2, 3],
              [4, 5, 7],
              [7, 8, 9]])

Y = np.array([[0],
              [1],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0]])



conv_1 = convolutional_layer.Conv_Layer((1, 28, 28), 2, (1, 2, 2))
conv_1_out = conv_1.forward(x_train[0])
print(np.shape(conv_1_out))
print("1", conv_1_out)
relu_1 = relu.ReLU()
relu_1_out = relu_1.forward(conv_1_out)
print(np.shape(relu_1_out))
maxl_1 = max_pooling_layer.max_pool_layer((2, 27, 27), (2, 2))
maxl_1_out = maxl_1.forward(relu_1_out)
print("shape of layer 1 ", np.shape(maxl_1_out))
print("2", maxl_1_out)
conv_2 = convolutional_layer.Conv_Layer((2, 13, 13), 3, (3, 5, 5))
conv_2_out = conv_2.forward(maxl_1_out)
print("shape of layer 2 before relu", np.shape(conv_2_out))
print("3", conv_2_out)
relu_2 = relu.ReLU()
relu_2_out = relu_2.forward(conv_2_out)
print("shape of layer 2  before max pooling", np.shape(relu_2_out))
maxl_2 = max_pooling_layer.max_pool_layer((3, 9, 9), (2, 2))
maxl_2_out = maxl_2.forward(relu_2_out)
print("shape of layer 2 before reshape_2 ", np.shape(maxl_2_out))
print(maxl_2_out)
reshape_2 = reshape.Reshape((3, 4, 4), (3 * 4 * 4, 1))
reshape_2_out = reshape_2.forward(maxl_2_out)

print("4", reshape_2_out)
print(np.shape(reshape_2_out))

layer_3 = dense.FC_layer(reshape_2_out, 50)
layer_3_out = layer_3.linear(True)
print("shape of layer_3_out before activation", np.shape(layer_3_out))
print("z in dense", layer_3.z)
print("weights in dense", layer_3.weights)
layer_3_out = layer_3.reLU(layer_3_out)

print(np.shape(layer_3_out))
layer_3_out = np.round_(layer_3_out, 4)
print("5", layer_3_out)

layer_4 = dense.FC_layer(layer_3_out, 10)
layer_4_out = layer_4.linear()
layer_4_out = layer_4_out
print("shape of layer 4 out before SoftMax, in main", np.shape(layer_4_out))
layer_4_out = layer_4.SoftMax(layer_4_out)
print("shape of layer 4 out, in main", np.shape(layer_4_out))
print("layer_4_out (prediction)", layer_4_out)
layer_4_out = np.where(layer_4_out == 1, layer_4_out - .000001, layer_4_out)

loss = loss.loss(Y)
print("error", loss.cross_entropy(layer_4_out))
dLdY = loss.cross_entropy_prime(layer_4_out)
print(np.shape(dLdY))
print(dLdY)

layer_4_d = layer_4.SoftMax(None, False, dLdY)
print("shape of layer 4 d before linear, in main", np.shape(layer_4_d))
layer_4_d = layer_4.linear(False, layer_4_d)  # should be the same size as layer_3_out

print("shape of layer 4 d, in main", np.shape(layer_4_d))

layer_3_d = layer_3.reLU(None, False, layer_4_d)
print("shape of layer 3 d before linear, in main ", np.shape(layer_3_d))
print("layer 3 d before layer 3", layer_3_d)
layer_3_d = layer_3.linear(False, layer_3_d)

print("shape of layer 3 d ", np.shape(layer_3_d))
print(layer_3_d)

layer_2_d = reshape_2.backward(layer_3_d)
print("shape of layer 2 d before max pooling", np.shape(layer_2_d))

layer_2_d = maxl_2.backward()
print("shape of layer 2 d before relu", np.shape(layer_2_d))
layer_2_d = relu_2.backward(layer_2_d)
print("shape of layer 2 d before conv layer", np.shape(layer_2_d))
layer_2_d = conv_2.backward(layer_2_d)
print("shape of layer 2 d", np.shape(layer_2_d))

layer_1_d = maxl_1.backward()
print("shape of layer 1 d before relu", np.shape(layer_1_d))

layer_1_d = relu_1.backward(layer_1_d)
print("shape of layer 1 d before conv layer", np.shape(layer_1_d))
print(np.shape(conv_1.input))
layer_1_d = conv_1.backward(layer_1_d)
print("shape of layer 1 d (input)", np.shape(layer_1_d))

right_guesses = 0
guesses = []
accuracies = []
errors = []


initial = np.copy(conv_2.filters)
e = 5
lr = .01
for i in range(e):
    conv_1_weights = np.copy(layer_4.weights)
    for l in range(np.shape(x_train)[0]):
        conv_1_out = conv_1.forward(x_train[l])
        relu_1_out = relu_1.forward(conv_1_out)
        maxl_1_out = maxl_1.forward(relu_1_out)
        conv_2_out = conv_2.forward(maxl_1_out)
        relu_2_out = relu_2.forward(conv_2_out)
        maxl_2_out = maxl_2.forward(relu_2_out)
        reshape_2_out = reshape_2.forward(maxl_2_out)
        layer_3_out = layer_3.linear(True)
        rshpe = np.copy(conv_2.filters)
        # print("z in dense", layer_3.z)
        # print("weights in dense", layer_3.weights)
        layer_3_out = layer_3.reLU(layer_3_out)
        layer_3_out = np.round_(layer_3_out, 0)
        layer_4 = dense.FC_layer(layer_3_out, 10)
        layer_4_out = layer_4.linear()
        layer_4_out = layer_4_out
        layer_4_out = layer_4.SoftMax(layer_4_out)

        print("guess, epoch:", i, "data point", l,  layer_4_out)
        layer_4_out = np.where(layer_4_out == 1, layer_4_out - .000001, layer_4_out)
        guesses.append(np.argmax(layer_4_out))
        # print("guess(index)", np.argmax(layer_4_out))
        # print("error", loss.binary_cross_entropy(layer_4_out))
        # print("accuracy(lower the better)", np.mean((layer_4_out-Y)**2))
        accuracies.append(np.mean((layer_4_out - Y) ** 2))
        errors.append(loss.cross_entropy(layer_4_out))
        if (np.argmax(layer_4_out) == np.argmax(Y)):
            right_guesses += 1

        dLdY = loss.cross_entropy_prime(layer_4_out)
        layer_4_d = layer_4.SoftMax(None, False, dLdY)
        layer_4_d = layer_4.linear(False, layer_4_d, lr)  # should be the same size as layer_3_out
        layer_3_d = layer_3.reLU(None, False, layer_4_d)
        # print("layer 3 d", layer_3_d)
        layer_3_d = layer_3.linear(False, layer_3_d, lr)
        # print("layer 3 d after linear", layer_3_d)
        layer_2_d = reshape_2.backward(layer_3_d)
        layer_2_d = maxl_2.backward()
        layer_2_d = relu_2.backward(layer_2_d)
        layer_2_d = conv_2.backward(layer_2_d, lr)
        layer_1_d = maxl_1.backward()
        layer_1_d = relu_1.backward(layer_1_d)
        layer_1_d = conv_1.backward(layer_1_d, lr)
        print("change in weights", conv_1_weights - layer_4.weights)

# print("right guesses", right_guesses)
print("percentage of guesses right", right_guesses / e)
print(accuracies)
print("errors", errors)
print("guesses", guesses)
print(rshpe[2])
print(initial[2])
print(np.amax(rshpe[0]))
print(initial[0]-rshpe[0])


f = plt.figure("x_train")
plt.imshow(x_train[-1][-1], cmap="gray")
g = plt.figure("rshpe")
plt.imshow(rshpe[0][0], cmap="gray")
h = plt.figure("rshpe2")
plt.imshow(rshpe[1][0], cmap="gray")
plt.show()


# print(layer_3.weights)
# print(conv_1_weights)
# print("change in weights", conv_1_weights-conv_1.filters)

# mean squared error
# seed:1, acc = .17451, loss = 1.2974 [[8.54134221e-05 1.96693470e-03 1.44271000e-03 2.89761719e-03 1.68453654e-03 9.72082115e-01 6.39127778e-03 1.32636301e-02 2.37834081e-05 1.61981723e-04]]
# seed:2, acc = .11956, loss = .11683 [[0.53257254 0.00750029 0.00812264 0.00187652 0.32153626 0.005410330.02981503 0.00540849 0.00106279 0.0866951 ]]
# seed:3, acc = .10031, loss = .19757 [[0.29905525 0.28281579 0.00576029 0.0271469  0.05572301 0.04542031 0.01988347 0.1045927  0.12295437 0.03664791]]

# print("accuracy(lower the better)", np.mean((layer_4_out-Y)**2))
