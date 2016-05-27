import tensorflow as tf
import math, random, numpy
import mlp

class EvoMLP():

    def __init__(self):
        topology = [
            (784, 'sigmoid'),
            (1, 'sigmoid'),
        ]
        self.network = mlp.MLP(topology)

    def forward(self, x):
        return self.network.apply(x)

    def train(self, x, y):
        self.network.train(x, y, 1)

        loss = 0.0
        output = self.network.apply(x)
        print(output)

        for i in range(len(x)):
            target = y[i][0]
            actual = output[i][0]
            loss += abs(target - actual)

        return loss / float(len(x))
