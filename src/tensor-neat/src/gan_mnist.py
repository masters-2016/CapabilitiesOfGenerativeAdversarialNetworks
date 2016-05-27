from math import log
import random
import sys

import matplotlib
matplotlib.use('Agg')

from neat import nn, population, statistics, visualize, config

from mnist import MNIST
mndata = MNIST('./data')
training_images,_ = mndata.load_testing() #training()

for i,image in enumerate(training_images):
    training_images[i] = [ x / 255. for x in image ]

eval_size = 10
image_width, image_height = 28,28
image_length = image_height * image_width

def dataset():
    return [random.choice(training_images) for x in range(eval_size)]

G_config = config.Config("mnist_config")
D_config = config.Config("mnist_config")

G_config.input_nodes = 12
G_config.output_nodes = 1
#G_config.report = False

D_config.input_nodes = image_length
D_config.output_nodes = 1
#D_config.report = False

def RMS(P,Q):
    """
    Calculate Binary Cross Entropy of two distributions, the "true" values P and
    "fake" values Q
    https://en.wikipedia.org/wiki/Cross_entropy
    """

    diff = 0
    for a,b in zip(P,Q):
        diff += (a - b)**2

    return diff / len(P)
    #return - sum(map(lambda x: x[0] * log(x[1] if x[1] != 0.0 else 0.00001),
    #            zip(P,Q)))

def apply_network(network, input_data):
    return network.serial_activate(input_data)

def evaluate_network(network, input_data, target_data):
    network_output = map(lambda x: apply_network(network,x),input_data)

    # TODO: Consider if this is weird / inadequate
    target_data = list(map(lambda x: sum(x) / len(x), target_data))
    network_output = list(map(lambda x: sum(x) / len(x), network_output))


    err = RMS(target_data,network_output)

    return err

def update_fitness(genomes):
    for g in genomes:
        input_data = dataset()
        output_data = [ (1.0,) for x in input_data ]

        input_data += [ [ random.random() for _ in range(image_length)] for _ in input_data ]
        output_data += [ (0.0,) for _ in input_data ]

        result = evaluate_network(  nn.create_feed_forward_phenotype(g),
                                    input_data, output_data )

        g.fitness = 1 - result

def update_fitness_G(genomes,D):

    for g in genomes:
        G = nn.create_feed_forward_phenotype(g)
        g.fitness = 0.0
        for i in range(eval_size):
            noise = [ random.random() for x in range(G_config.input_nodes - 2) ]

            generated =  [ apply_network(G,noise + [x, y])[0]
                            for x in range(28)
                                for y in range(28) ]

            g.fitness += apply_network(D,generated)[0] / eval_size


def update_fitness_D(genomes,G):

    fake_data = []
    for i in range(eval_size):

        noise = [ random.random() for x in range(G_config.input_nodes - 2) ]

        generated =  [ apply_network(G,noise + [x, y])[0]
                        for x in range(28)
                            for y in range(28) ]

        fake_data.append(generated)

    real_data = dataset()

    inputs = real_data + fake_data
    outputs = [ [1] for x in real_data ] + [ [0] for x in fake_data ]

    for g in genomes:

        g.fitness = 1 - evaluate_network(nn.create_feed_forward_phenotype(g),
                                        inputs, outputs)

def get_evo_network(pop):

    winner = pop.statistics.best_genome()
    return nn.create_feed_forward_phenotype(winner)

if __name__ == "__main__":

    # Bootstrapping networks
    G_pop = population.Population(G_config)
    print("G bootstrapped")
    D_pop = population.Population(D_config)
    print("D bootstrapped")

    D_pop.run(update_fitness,1)
    G_pop.run(lambda gs: update_fitness_G(gs,get_evo_network(D_pop)),1)

    for e in range(int(sys.argv[1])): # Epochs
        D_pop.run(lambda gs: update_fitness_D(gs,get_evo_network(G_pop)),5)
        print("D best", D_pop.statistics.best_genome().fitness)

        G_pop.run(lambda gs: update_fitness_G(gs,get_evo_network(D_pop)),5)
        print("G best", G_pop.statistics.best_genome().fitness)

        fake_G_data = [ random.random() for x in range(G_config.input_nodes) ]
        fake_D_data = [ random.random() for x in range(D_config.input_nodes) ]
        print("G output", apply_network(get_evo_network(G_pop),fake_G_data))
        print("D output", apply_network(get_evo_network(D_pop),fake_D_data))

    D = get_evo_network(D_pop)
    G = get_evo_network(G_pop)

    for input_data,output_data in dataset():
        fake_data = [ random.random() for x in range(D_config.input_nodes) ]
        print("D on {}: {}".format(input_data,apply_network(D,fake_data)))
        print("D on {}: {}".format(fake_data,apply_network(D,input_data)))

    for _ in dataset():
        input_data = [ random.random() for _ in range(G_config.input_nodes) ]
        print("G on {}: {}".format(input_data,apply_network(G,input_data)))
