from dataset import load_dataset
import pickle

def dump(dataset):
    print ("dumping",dataset)
    with open(dataset+".pickle","wb") as f:
        pickle.dump(load_dataset(dataset),f)

def load(dataset):
    with open(dataset+".pickle","rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    #dump('../data/test')
    #dump('../data/validation')
    #dump('../data/unmodified')
    dump('data/random_crop')
    dump('data/gan_generated')
    dump('data/unmodified_plus_gan_generated')
