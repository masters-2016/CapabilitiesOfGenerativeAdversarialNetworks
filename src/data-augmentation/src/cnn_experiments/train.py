from cnn import CNN
from compress import load
import sys, time
import random
import numpy as np

if __name__ == "__main__":

    epochs = int(sys.argv[1])
    dataset = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    print("Loading training data %s..." % dataset)

    x_train, y_train = load(dataset)
    print("Loading validation data ...")
    x_validation, y_validation = load("validation")
    print("Loading test data ...")
    x_test, y_test = load("test")

    print('Zipping')
    train_zipped = list(zip(x_train,y_train))

    print("Shuffling")
    random.shuffle(train_zipped)

    print("Rebuilding x_train")
    x_train = np.array([ e[0] for e in train_zipped])

    print("Rebuilding y_train")
    y_train = np.array([ e[1] for e in train_zipped])

    print("Creating CNN")
    cnn = CNN()

    total_start = time.time()

    print("Training for %d epochs in batchsize %d" % (epochs,batch_size))
    batch_count = len(x_train) / batch_size
    for epoch in range(epochs):
        epoch_start = time.time()

        print("Doing epoch %d" % epoch)

        train_batches = 0
        accum_training_loss = 0
        for x in range(0,len(x_train), batch_size):
            subepoch_time = time.time()

            train_batches += 1
            training_loss = cnn.train( x_train[x:x+batch_size],
                                        y_train[x:x+batch_size])
            accum_training_loss += training_loss

            print("\tSubepoch %06d / %06d : %.6f (%f)" % (train_batches, batch_count, training_loss, time.time() - subepoch_time))

        training_time = time.time() - epoch_start

        validation_batches = 0
        accum_validation_loss = 0
        for x in range(0,len(x_validation), batch_size):
            validation_batches += 1
            accum_validation_loss += cnn.validate(  x_validation[x:x+batch_size],
                                                    y_validation[x:x+batch_size])

        validation_time = time.time() - epoch_start

        total_time = time.time() - total_start

        print('Epoch %06d %.6f %.6f %f %f %f' % (  epoch,
                                    accum_training_loss / train_batches,
                                    accum_validation_loss / validation_batches,
                                    training_time, validation_time, total_time))

    test_loss = cnn.validate(x_test, y_test)
    print('Test loss: %.6f' % test_loss)
