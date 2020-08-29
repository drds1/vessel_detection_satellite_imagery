#image analysis project from vtx
#add training dataset from kaggle project
#https://www.kaggle.com/rhammell/ships-in-satellite-imagery
#paper https://www.irjet.net/archives/V6/i9/IRJET-V6I9291.pdf

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import glob
import json
from sklearn.model_selection import train_test_split
import keras

def get_label_from_filename(data_dir = './data/images/shipsnet/shipsnet/'):
    '''

    :return:
    '''
    #list all files image files in dataset
    files = glob.glob(data_dir+'*.png')
    files = [f.replace(data_dir,'') for f in files]

    #the target is given in the image file name by the prefix 1 or 0
    #extract this information
    target = np.array([int(f[0]) for f in files])
    print('target info extracted')
    for t in np.unique(target):
        print('found '+str(len(target[target==t]))+'images with class label '+str(t))


def load_data_from_json(file = './data/shipsnet.json'):
    '''

    :return:
    '''
    # download dataset from json object
    f = open(file)
    dataset = json.load(f)
    f.close()

    # load input and output data
    input_data = np.array(dataset['data']).astype('uint8')
    output_data = np.array(dataset['labels']).astype('uint8')

    # input data has been flattened in the json but
    # is actually a 80x80x3 image. reshape here
    n_spectrum = 3  # color chanel (RGB)
    weight = 80
    height = 80
    X = input_data.reshape([-1, n_spectrum, weight, height])
    return X, output_data



def plot_example_rgb(pic, savefile=False):
    '''
    input example X image pic[X,Y,z]
    :return:
    '''

    # get one chanel
    #pic = X[0]
    plt.close()
    rad_spectrum = pic[0]
    green_spectrum = pic[1]
    blue_spectum = pic[2]

    plt.figure(2, figsize=(5 * 3, 5 * 1))
    plt.set_cmap('jet')

    # show each channel
    plt.subplot(1, 3, 1)
    plt.imshow(rad_spectrum)

    plt.subplot(1, 3, 2)
    plt.imshow(green_spectrum)

    plt.subplot(1, 3, 3)
    plt.imshow(blue_spectum)
    if savefile is False:
        plt.show()
    else:
        plt.savefig(savefile)
        plt.close()



def define_custom_convnet():
    '''
    convnet with 4 sequential conv, max-pool, dropout layers
    :return:
    '''

    # network design
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 40x40
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 20x20
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 10x10
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (10, 10), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))  # 5x5
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2, activation='softmax'))

    # optimization setup
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])

    return model



if __name__ == '__main__':

    # download dataset from json object
    X, y = load_data_from_json(file = './data/shipsnet.json')

    # plot image file
    plot_example_rgb(X[0], savefile='example_input.png')

    # split into train test data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.33,
                                                        random_state = 42)

    # define the model
    model = define_custom_convnet()

    # fit the model
    model.fit(X_train, y_train)









