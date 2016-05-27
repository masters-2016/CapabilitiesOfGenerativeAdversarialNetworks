import numpy as np

from scipy import misc

def load_dataset(dirname):
    with open('%s/labels.txt' % dirname) as f:
        images = []
        labels = []

        for line in f:
            parts = line[:-1].split(' ')
            img = [ x / 255 for x in misc.imread('%s/%s' % (dirname, parts[0])).flatten()]

            lab = [ float(x) for x in parts[1].split(',') ]

            images.append(np.array(img))
            labels.append(lab)

        return images, labels
