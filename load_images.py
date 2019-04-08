import numpy as np
from PIL import Image

class KaggleImageLoader:

    def LoadImage():
        path_train_images = './Data/Train/TrainImages/'
        path_validation_images = './Data/Validation/ValidationImages/'

        x_train = [] # Empty list
        for i in range(1,5831):
            img = Image.open(path_train_images + 'Image' + str(i) + '.jpg') # Loading Image<x>.jpg
            train.append(np.array(img))
        x_train = np.asarray(x_train)
        y_train = pd.read_csv('./Data/Train/trainLbls.csv',header=None)

        x_test = [] # Empty list
        for i in range(1,2299):
            img = Image.open(path_validation_images + 'Image' + str(i) + '.jpg') # Loading Image<x>.jpg
            validation.append(np.array(img))
        x_text = np.asarray(validation)
        y_test = pd.read_csv('./Data/Validation/valLbls.csv',header=None)

        return x_train, y_train, x_test, y_test


    
    def LoadShuffledSubset():
        x_train, y_train, x_test, y_test = LoadImage()

        train_set = np.concatenate((x_train,y_train), axis=1)
        np.random.shuffle(train_set)
        train = train_set[:,0:4096]
        train_lbls = train_set[:,-1]

        