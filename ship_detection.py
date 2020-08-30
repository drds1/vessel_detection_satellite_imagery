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
import pickle
import os
from PIL import Image
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.applications.resnet50 import ResNet50

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
    X = np.transpose(X, (0, 2, 3, 1))
    return X, output_data



def plot_example_rgb(pic, savefile=None):
    '''
    input example X image pic[X,Y,z]
    :return:
    '''

    # get one chanel
    #pic = X[0]
    plt.close()
    rad_spectrum = pic[:,:,0]
    green_spectrum = pic[:,:,1]
    blue_spectum = pic[:,:,2]

    plt.figure(2, figsize=(5 * 3, 5 * 1))
    plt.set_cmap('jet')

    # show each channel
    plt.subplot(1, 3, 1)
    plt.imshow(rad_spectrum)

    plt.subplot(1, 3, 2)
    plt.imshow(green_spectrum)

    plt.subplot(1, 3, 3)
    plt.imshow(blue_spectum)
    if savefile is None:
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


def define_resnet_model():
    '''

    :return:
    '''
    base_model = ResNet50(weights='imagenet')
    base_model.trainable = False
    output_model = keras.Sequential()
    output_model.add(keras.layers.Dense(32, activation='relu'))
    output_model.add(keras.layers.Dense(2, activation='softmax'))
    tl_model = keras.Sequential()
    tl_model.add(base_model)
    tl_model.add(output_model)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    tl_model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])
    return tl_model



def convert_image_dimensions(X_train_norm, newsize=(224,224)):
    '''
    resize image using PIL (resnet 50 needs 224 by 224)
    :param X_train_norm:
    :param newsize:
    :return:
    '''
    N = np.shape(X_train_norm)[0]
    X_train_norm_resize = np.zeros((N,newsize[0],newsize[1],3))
    for i in range(N):
        x = np.array(X_train_norm[i, :, :, :])
        xn = np.uint8( x / x.max()* 255 )
        img = Image.fromarray(xn, mode='RGB')
        img2 = img.resize(newsize, Image.ANTIALIAS)
        X_train_norm_resize[i,:,:,:] = np.array(img2)
    return X_train_norm_resize

def diagnostic_plots(y_pred_probs,y_test_probs,
                     labels_in = None,
                     diagnostic_file = 'roc_plot.png',
                     max_fpr_tollerance = 0.01):
    '''

    :return:
    '''
    # fit model on test data
    fpr, tpr, thresholds = metrics.roc_curve(y_test_probs[:, 0], y_pred_probs[:, 0], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    idx = np.argsort(fpr)
    fpr = fpr[idx]
    tpr = tpr[idx]
    thresholds = thresholds[idx]
    idx_threshold = np.where(fpr > max_fpr_tollerance)[0][0]
    threshold_tollerance = thresholds[idx_threshold]
    print('ROC curve AUC = ' + str(auc))
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ann = 'threshold probability P(FPR=' + str(max_fpr_tollerance) + ') = ' \
          + str(np.round(threshold_tollerance, 2)) + \
          '\n TPR = ' + str(np.round(tpr[idx_threshold], 2))
    ax1.axvline(fpr[idx_threshold], label=ann, color='b')
    ax1.plot(fpr, tpr, label='AUC = ' + str(np.round(auc, 2)), color='r')
    ax1.set_title('model ROC curve')
    ax1.legend()

    # now add confusion matrix for tollerance result
    cm = metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    cmn = cm / cm.sum(axis=1)[:, np.newaxis]
    ax2 = fig.add_subplot(2, 2, 3)
    b = ax2.imshow(cmn, cmap='Blues')
    cbar = fig.colorbar(b)
    cbar.set_label('Normalised Counts')
    if labels_in is None:
        labels = list(np.array(np.unique(y_test), dtype=str))
    else:
        labels = labels_in
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix \nP(FPR=' + str(max_fpr_tollerance) + ')')
    ax2.set_xticks = np.arange(len(labels))
    ax2.set_xticklabels = labels
    ax2.set_yticks = np.arange(len(labels))
    ax2.set_yticklabels = labels
    plt.tight_layout()
    plt.savefig(diagnostic_file)


def fit_load_model(X_train_norm, y_train,
                   new_model=False,
                   picklefile='./models/custom_convnet.pickle',
                   input_model=None):
    '''
    fit or load a new model from file
    :return:
    '''
    if new_model is True:
        model = input_model
        # fit the model
        model.fit(X_train_norm, y_train,
                  batch_size=32,
                  epochs=18,
                  validation_split=0.2,
                  shuffle=True,
                  verbose=2)
        # pickle the fitted model
        os.system('rm ' + picklefile)
        pickle_out = open(picklefile, "wb")
        pickle.dump({'model': model}, pickle_out)
        pickle_out.close()
    else:
        model = pickle.load(open(picklefile, "rb"))['model']
    return model


if __name__ == '__main__':

    # download dataset from json object
    X, target = load_data_from_json(file = './data/shipsnet.json')

    # the target is a 2d array of the probability
    # that the image is a ship or isnt a ship
    # need to construct this 2d prob array
    y = keras.utils.np_utils.to_categorical(target)

    # plot image file
    plot_example_rgb(X[0], savefile = 'example_input.png')

    # split into train test data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.33,
                                                        random_state = 42)

    #transform by dividing through by minimum value
    norm = X_train.max()
    X_train_norm = X_train/norm
    X_test_norm = X_test/norm

    # define the model
    model = define_custom_convnet()
    model = fit_load_model(X_train_norm, y_train,
                              new_model=False,
                              picklefile='./models/custom_convnet.pickle',
                              input_model=model)

    # score model on test data
    y_pred = model.predict(X_test_norm)
    diagnostic_plots(y_pred, y_test,
                     labels_in=None,
                     diagnostic_file='roc_plot.png',
                     max_fpr_tollerance=0.01)




    #now try resnet 50 transfer learning model
    #resnet 50 needs image sizes to be 224 x 224
    #use python image library PIL to resize
    X_train_norm_resize = convert_image_dimensions(X_train_norm, newsize=(224, 224))
    plot_example_rgb(X_train_norm[0,:,:,:], savefile='normed_image_example.png')
    plot_example_rgb(X_train_norm_resize[0, :, :, :], savefile='normed_resized_image_example.png')

    #assemble model using transfer learning approach using resnet50 and output sequential mode
    tl_model = define_resnet_model()
    tl_model = fit_load_model(X_train_norm_resize,y_train,
                              new_model=True,
                              picklefile ='./models/tl_resnet.pickle',
                              input_model= tl_model)

    # fit model on test data
    y_pred_tl = tl_model.predict(X_test_norm)
    diagnostic_plots(y_pred_tl, y_test,
                     labels_in=None,
                     diagnostic_file='roc_plot_tl.png',
                     max_fpr_tollerance=0.01)














