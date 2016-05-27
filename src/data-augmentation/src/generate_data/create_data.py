from PIL import Image
import numpy, random

def save_image(image, filename):
    '''
    Saves the given image to a file at the given path

    image: list (list float)
    filename: string
    returns: None
    '''
    img_dim = len(image[0])

    frame = Image.new('L', (img_dim, img_dim), (255,))

    pixels = list( range(img_dim) )

    for x in pixels:
        for y in pixels:
            value = max(0, min(255, int( image[x, y]*255 )))
            frame.putpixel( (x, y), value )

    frame.save(filename, "PNG")


def array_to_image(arr, img_dim=28):
        image = []
        for j in range(img_dim):
            image.append( numpy.array( arr[j*img_dim:j*img_dim+img_dim] ) )

        image = numpy.array(image)
        image = numpy.rot90(image, 3)
        image = numpy.fliplr(image)

        return image


def save_unmodified(mnist):
    with open("unmodified/labels.txt","w") as labels_unmodified:
        for i in range(len(mnist.train.images)):
            label_str = ",".join(map (lambda x: str(int(x)),mnist.train.labels[i]))
            #save_image(array_to_image(mnist.train.images[i]),"unmodified/%06d.png" % i)
            print("%06d.png %s" % (i,label_str),file=labels_unmodified)

def save_random_crop(mnist):
    with open("random_crop/labels.txt","w") as labels:
        x = 0
        for i in range(len(mnist.train.images)):
            img = array_to_image(mnist.train.images[i])
            label_str = ",".join(map(lambda x: str(int(x)), mnist.train.labels[i]))

            padded = numpy.pad(img, 4, mode='constant', constant_values=0)


            for j in range(4):
                start_x = random.randint(0, 7)
                start_y = random.randint(0, 7)

                cropped = padded[start_x:start_x+28, start_y:start_y+28]

                save_image(cropped, "random_crop/%06d.png" % x)
                print("%06d.png %s" % (x, label_str), file=labels)

                x += 1


def save_test(mnist):
    with open("test/labels.txt","w") as labels:
        for i in range(len(mnist.test.images)):
            label_str = ",".join(map(lambda x: str(int(x)), mnist.train.labels[i]))
            print("%06d.png %s" % (i, label_str), file=labels)

            save_image(array_to_image(mnist.test.images[i]), "test/%06d.png" % i)

def save_validation(mnist):
    with open("validation/labels.txt","w") as labels:
        for i in range(len(mnist.validation.images)):
            label_str = ",".join(map(lambda x: str(int(x)), mnist.train.labels[i]))
            print("%06d.png %s" % (i, label_str), file=labels)

            save_image(array_to_image(mnist.validation.images[i]), "validation/%06d.png" % i)


if __name__ == "__main__":
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    #save_unmodified(mnist)
    save_random_crop(mnist)
    #save_test(mnist)
    #save_validation(mnist)
