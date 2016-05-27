import numpy, PIL, neat, random

def generate_image(genome, img_dim=28):
    '''
    Generate an image from the given genom

    genome: genome. From neat-python
    returns: list (list float)
    '''
    network = neat.nn.create_feed_forward_phenotype(genome)

    image = numpy.empty([img_dim, img_dim])
    noise = [ random.random() for x in range(10) ]

    pixels = list( range(img_dim) )
    steps = [ x / ((img_dim-1)/2) - 1 for x in pixels ]

    for x in pixels:
        for y in pixels:
            input_data = numpy.array([ steps[x], steps[y] ] + noise)
            output = network.serial_activate(input_data)
            image[x, y] = output[0]

    return image

def save_image(image, filename):
    '''
    Saves the given image to a file at the given path

    image: list (list float)
    filename: string
    returns: None
    '''
    img_dim = len(image[0])

    frame = PIL.Image.new('L', (img_dim, img_dim), (255,))

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
