from math import log
import random
import sys

import matplotlib
matplotlib.use('Agg')

from neat import nn, population, statistics, visualize, config

G_config = config.Config("ones_config")
D_config = config.Config("ones_config")

G_config.input_nodes = 4
G_config.output_nodes = 4
G_config.report = False

D_config.input_nodes = G_config.output_nodes
D_config.output_nodes = 1
D_config.report = False


# Network inputs and expected outputs.
dataset = [
    ([1 for x in range(G_config.output_nodes)],[1]),
]

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
        input_data,output_data = zip(*dataset)
        result = evaluate_network(  nn.create_feed_forward_phenotype(g),
                                    input_data, output_data )
        g.fitness = 1 - result

def update_fitness_G(genomes,D):

    for g in genomes:

        g.fitness = 0.0
        for i in range(len(dataset)):
            noise = [ random.random() for x in range(G_config.input_nodes) ]
            generated = apply_network(nn.create_feed_forward_phenotype(g),noise)
            g.fitness += apply_network(D,generated)[0] / len(dataset)


def update_fitness_D(genomes,G):

    for g in genomes:
        real_inputs, real_outputs = zip(*dataset)

        noise = [ random.random() for x in range(G_config.input_nodes) ]

        fake_inputs = [ apply_network(G,noise) for _ in range( len(dataset) ) ]
        fake_outputs = [ [0] for x in range( len(dataset) ) ]

        inputs = list(real_inputs) + fake_inputs
        outputs = list(real_outputs) + fake_outputs


        g.fitness = 1 - evaluate_network(nn.create_feed_forward_phenotype(g),
                                        inputs,outputs)

def get_evo_network(pop):

    winner = pop.statistics.best_genome()
    return nn.create_feed_forward_phenotype(winner)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print(RMS([1,1],[1,1]))
        print(RMS([1,1],[0.001,1]))
        print(RMS([1,1],[0.001,000000.1]))
    else:
        # Bootstrapping networks
        G_pop = population.Population(G_config)
        D_pop = population.Population(D_config)

        D_pop.run(update_fitness,1)
        G_pop.run(lambda gs: update_fitness_G(gs,get_evo_network(D_pop)),1)

        for e in range(int(sys.argv[1])): # Epochs
            D_pop.run(lambda gs: update_fitness_D(gs,get_evo_network(G_pop)),5)
            G_pop.run(lambda gs: update_fitness_G(gs,get_evo_network(D_pop)),5)

            print("D best", D_pop.statistics.best_genome().fitness)
            print("G best", G_pop.statistics.best_genome().fitness)

            fake_data = [ random.random() for x in range(G_config.input_nodes) ]
            print("G output", apply_network(get_evo_network(G_pop),fake_data))
            print("D output", apply_network(get_evo_network(D_pop),fake_data))

        D = get_evo_network(D_pop)
        G = get_evo_network(G_pop)

        for input_data,output_data in dataset:
            fake_data = [ random.random() for x in range(D_config.input_nodes) ]
            print("D on {}: {}".format(input_data,apply_network(D,fake_data)))
            print("D on {}: {}".format(fake_data,apply_network(D,input_data)))

        for _ in dataset:
            input_data = [ random.random() for _ in range(G_config.output_nodes) ]
            print("G on {}: {}".format(input_data,apply_network(G,input_data)))
