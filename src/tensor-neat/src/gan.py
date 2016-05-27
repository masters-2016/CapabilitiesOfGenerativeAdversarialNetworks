import pybrain_mlp
import cnn, evo_mlp, cppn, util
import numpy, sys, random

from tensorflow.examples.tutorials.mnist import input_data

def random_batch(x, y, batch_size):
    xr = []
    yr = []

    for i in range(batch_size):
        idx = random.randint(0, len(x)-1)
        xr.append(x[idx])
        yr.append(y[idx])

    return numpy.array(xr), numpy.array(yr)

def log(s):
    print(s)
    sys.stdout.flush()

if __name__ == '__main__':
    log('Loading dataset ...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    real_x = mnist.train.images[:1]
    real_y = numpy.array([ numpy.array([1.0]) for x in real_x ])

    for i in range(1):
        image = util.array_to_image(real_x[i])
        util.save_image(image, 'out/real_%d.png' % i)

    log('Constructing D ...')
    #d = cnn.CNN()
    #d = evo_mlp.EvoMLP()
    d = pybrain_mlp.PyBrainMLP([784, 128, 1])

    log('Constructing G ...')
    def fitness_func(generator, genome):
        x = numpy.array([ util.generate_image(genome).flatten() for x in range(5)])
        y =  [ v[0] for v in d.forward(x) ]
        return sum(y) / float( len(y) )

    g = cppn.CPPN('mnist_config', fitness_func)

    log('Initializing G ...')
    g.train()


    epochs = 10000
    batches = 100
    generations = 1

    for e in range(epochs):
        log('Epoch %d' % e)

        #log('\tD sanity check ...')
        #x = numpy.array([ g.generate_image().flatten() for x in range(3) ])
        #y =  [ v[0] for v in d.forward(x) ]
        #log('\t\t Fake data: %s' % str(y))
        #x = numpy.array([ random.choice(real_x) for x in range(3) ])
        #y =  [ v[0] for v in d.forward(x) ]
        #log('\t\t Real data: %s' % str(y))

        log('\tTraining D ...')
        for b in range(batches):
            fake_batch_x = numpy.array([ img.flatten() for img in g.generate_images_for_all_genomes()])
            fake_batch_y = numpy.array([ numpy.array([0.0]) for x in fake_batch_x ])

            real_batch_x, real_batch_y = random_batch(real_x, real_y, len(fake_batch_x))


            batch_x = numpy.concatenate( (real_batch_x, fake_batch_x) )
            batch_y = numpy.concatenate( (real_batch_y, fake_batch_y) )

            loss = d.train(batch_x, batch_y)

            log('\t\t %d / %d -> loss = %f' % (b+1, batches, loss))

            if loss < 0.05:
                log('\t\tContinuing as D is sufficiently good')
                break

        #if loss > 0.4:
        #    log('\tTerminating, as D can no longer detect fakes')
        #    break

        #log('\tD sanity check ...')
        #x = numpy.array([ g.generate_image().flatten() for x in range(3) ])
        #y =  [ v[0] for v in d.forward(x) ]
        #log('\t\t Fake data: %s' % str(y))
        #x = numpy.array([ random.choice(real_x) for x in range(3) ])
        #y =  [ v[0] for v in d.forward(x) ]
        #log('\t\t Real data: %s' % str(y))

        
        log('\tTraining G ...')
        for b in range(generations):
            fitness = g.train()
            log('\t\t %d / %d -> fitness = %f' % (b+1, generations, fitness))
            #if fitness > 0.95:
            #    log('\t\tContinuing as G is sufficiently good')
            #    break

        image = g.generate_image()
        util.save_image(image, 'out/%d.png' % e)

    for i in range(50):
        image = g.generate_image()
        util.save_image(image, 'out/result_%d.png' % i)
