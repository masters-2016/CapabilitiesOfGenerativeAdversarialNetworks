import util
import matplotlib, numpy, PIL
matplotlib.use('Agg')

from neat import nn, population, statistics, visualize, config

import random

class CPPN():

    def __init__(self, cfg, fitness_func, img_dim=28):
        '''
        Create a CPPN from the given config and fitness function

        cgf: string or neat.config
        fitness_func: genomes -> float
        returns: CPPN
        '''
        if type(cfg) == str:
            cfg = config.Config(cfg)
        cfg.report = False

        self.img_dim = img_dim

        self.pop = population.Population(cfg)

        def wrapped_fitness_func(genomes):
            for g in genomes:
                g.fitness = fitness_func(self, g)

        self.fitness_func = wrapped_fitness_func

    def train(self, generations=1):
        '''
        Train the CPPN for the given number of generations.

        generations: int. The number of generations to train
        returns: float. The fitness of the best genome after training
        '''
        self.pop.run(self.fitness_func, generations)
        return self.pop.statistics.best_genome().fitness

    def generate_image(self):
        '''
        Generate an image with the CPPN

        returns: list (list float)
        '''
        genome = self.pop.statistics.best_genome()
        return util.generate_image(genome)

    def generate_images_for_all_genomes(self):
        '''
        Generate an image with the CPPN for each genome

        returns: list (list (list float))
        '''
        images = []
        for specie in self.pop.species:
            for genome in specie.members:
                images.append(util.generate_image(genome))
        return images
    
    def save_stats(self, file_prefix):
        '''
        Save info about the network to a number of files with the given file prefix

        file_prefix: string
        returns: None
        '''
        best_genome = self.pop.statistics.best_genome()

        visualize.plot_stats(self.pop.statistics, filename="%s_stats.png" % file_prefix)
        visualize.plot_species(self.pop.statistics, filename="%s_stats.png" % file_prefix)
        visualize.draw_net(best_genome, view=False, filename="%s_stats.gv" % file_prefix)
        statistics.save_stats(self.pop.statistics, filename="%s_stats.csv" % file_prefix)
        statistics.save_species_count(self.pop.statistics, filename="%s_species_count.csv" % file_prefix)
        statistics.save_species_fitness(self.pop.statistics, filename="%s_species_fitness.csv" % file_prefix)


if __name__ == '__main__':
    import random

    def fitness_func(cppn, genomes):
        return [ random.random() for x in genomes ]

    cppn = CPPN('mnist_config', fitness_func)
    cppn.train(3)
    cppn.generate_image_file('out/test.png')
