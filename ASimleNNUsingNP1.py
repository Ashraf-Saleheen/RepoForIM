import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight) + self.bias

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_losses = np.mean(sample_losses)
        return data_losses
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) ==1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape)==2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood
X,y = spiral_data(samples=100, classes=3)

dense1 = Dense_layer(2,3)
activation1 = Activation_ReLU()
dense2 = Dense_layer(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss : ", loss)


