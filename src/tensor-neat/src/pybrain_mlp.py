from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import util

class PyBrainMLP():

    def __init__(self, topology):
        self.network = buildNetwork(*topology, bias=True, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)

    def forward(self, x):
        return [ self.network.activate(v) for v in x ]

    def train(self, x, y, learning_rate=0.01):
        ds = SupervisedDataSet( len(x[0]), len(y[0]) )

        util.save_image(util.array_to_image(x[0]), 'out/real.png')
        util.save_image(util.array_to_image(x[-1]), 'out/fake.png')

        for i in range( len(x) ):
            ds.addSample( x[i], y[i] )

        trainer = BackpropTrainer(self.network, ds, learningrate=learning_rate)
        return trainer.train()


if __name__ == '__main__':
    #input_data = [[0.], [1.]]
    #output_data = [[1.], [0.]]

    #mlp = PyBrainMLP([(1, 'none'), (5, 'sigmoid'), (1, 'sigmoid')])
    #mlp.train(input_data, output_data, 10000, print_interval=1000)

    input_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    output_data = [[0.], [1.], [1.], [0.]]

    mlp = PyBrainMLP([2, 5, 1])

    for i in range(1000):
        print("%d: %f" % (i, mlp.train(input_data, output_data)))

    for i in range(len(input_data)):
        inp = input_data[i]
        result = mlp.forward( inp )
        print('%s -> %s' % (inp, result))
