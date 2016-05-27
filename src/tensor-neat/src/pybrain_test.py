from pybrain_mlp import PyBrainMLP
from tensorflow.examples.tutorials.mnist import input_data

import util

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    real = mnist.train.images[0]
    fake = [ 0.0 for x in real ]

    util.save_image(util.array_to_image(real), 'out/real.png')
    util.save_image(util.array_to_image(fake), 'out/fake.png')

    x = [ real, fake ]
    y = [ [1.0], [0.0] ]

    mlp = PyBrainMLP([784, 1])

    for i in range(10000):
        loss = mlp.train(x, y)

        if i % 100 == 0:
            print("%d: %f" % (i, loss))

        if i % 100 == 0:
            result = mlp.forward(x)
            print('Real: %s' % result[0])
            print('Fake: %s' % result[1])
