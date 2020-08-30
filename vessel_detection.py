#image analysis project from vtx
#add training dataset from kaggle project
#https://www.kaggle.com/rhammell/ships-in-satellite-imagery
#paper https://www.irjet.net/archives/V6/i9/IRJET-V6I9291.pdf

import numpy as np
import matplotlib.pylab as plt
import json
from sklearn.model_selection import train_test_split
import keras
import pickle
import os
from sklearn import metrics
from PIL import Image
from keras_sequential_ascii import keras2ascii

def load_data_from_json(file = './data/shipsnet.json'):
    '''
    load input data and target labels from json at location specified by
    file argument.
    :return:
    X[N, img_width, img_height, color channels]: image pixels
    target[N]: 1 / 0 labels
    :return:
    '''
    # download dataset from json object
    f = open(file)
    data = json.load(f)
    f.close()

    # Ingest images and labels
    input_data = np.array(data['data']).astype('uint8')
    output_data = np.array(data['labels']).astype('uint8')

    # Json input images are all 1D. Need to convert into (80, 80, 3)
    # final 3rd dim colour channel (RGB)
    X = input_data.reshape([-1, 3, 80, 80])
    # reorder indices for required keras input format
    X = np.transpose(X, (0, 2, 3, 1))
    return X, output_data


def plot_example_rgb(pic, savefile=None):
    '''
    Plot all channels of input image
    :return:
    '''
    plt.close()
    labels = ['Red','Greed','Blue']
    fig = plt.figure()
    for i in range(np.shape(pic)[2]):
        pixels = pic[:,:,i]
        ax1 = fig.add_subplot(3,1,i+1)
        ax1.imshow(pixels,cmap='Blues')
        ax1.set_title(labels[i])
    plt.tight_layout()
    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        plt.close()



def conv_pool_dropout_block(modelin,
                            input_shape = None,
                            Nfilters = 32,
                            filtersize = (3, 3),
                            poolsize = (2, 2),
                            padding = 'same',
                            activation = 'relu',
                            dropout_frac = 0.3):
    '''
    Sequential conv, pooling and dropout layers tend to perform well
    in classification problems
    :param modelin:
    :param filtersize:
    :param poolsize:
    :param padding:
    :return:
    '''
    model = modelin
    if input_shape is not None:
        model.add(keras.layers.Conv2D(Nfilters, filtersize,
                                      padding=padding,
                                      input_shape=input_shape,
                                      activation=activation))
    else:
        model.add(keras.layers.Conv2D(Nfilters, filtersize,
                                      padding=padding,
                                      activation=activation))
    model.add(keras.layers.MaxPooling2D(pool_size=poolsize))  # 40x40
    model.add(keras.layers.Dropout(dropout_frac))
    return model


def define_custom_convnet(image_dim = (80,80,3)):
    '''
    convnet with 4 sequential conv, max-pool, dropout layers
    :return:
    '''

    # network design
    model = keras.Sequential()

    #apply 4 conv blocks
    for i in range(4):
        if i == 0:
            input_shape = image_dim
        else:
            input_shape = None
        model = conv_pool_dropout_block(model,
                                input_shape = input_shape,
                                Nfilters=32,
                                filtersize=(3, 3),
                                poolsize=(2, 2),
                                padding='same',
                                activation='relu',
                                dropout_frac=0.3)

    #apply ouptput classifier layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))

    # optimization setup
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy'])
    return model


def diagnostic_plots(y_pred_probs,y_test_probs,
                     labels_in = None,
                     diagnostic_file = './images/roc_plot.png',
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


def plot_example_classes(X, y_pred=None, y_test=None, N_examples = 3, filename='./images/examples.png',
                         idx_custom = None, labels_custom = None):
    '''
    show some example classifications demonstrating where the CNN
    got it both right and wrong
    :param y_pred:
    :param y_test:
    :param N_examples:
    :param filename:
    :return:
    '''
    if y_pred is None and idx_custom is None:
        raise Exception('one of y_pred or idx_custom must not be None')
    if y_pred is not None:
        pred_class = np.argmax(y_pred, axis=1)
    if y_test is not None:
        test_class = np.argmax(y_test, axis=1)
    if idx_custom is None:
        idx_FP = np.where((pred_class == 1) & (test_class == 0))[0]
        idx_FN = np.where((pred_class == 0) & (test_class == 1))[0]
        idx_TP = np.where((pred_class == 1) & (test_class == 1))[0]
        idx_TN = np.where((pred_class == 0) & (test_class == 0))[0]
        idx_groups = [idx_TP, idx_TN, idx_FP, idx_FN]
    else:
        idx_groups = idx_custom
    if labels_custom is None:
        labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
    else:
        labels = labels_custom
    i = 0
    fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=len(labels), ncols=1, sharey=True)
    itemp = 0
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title(labels[itemp], fontsize=24)
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False
        itemp += 1
    for idx in idx_groups:
        n = min(len(idx), N_examples)
        for i2 in range(n):
            ax1 = fig.add_subplot(len(idx_groups), N_examples, i * N_examples + i2 + 1)
            x = np.array(X[idx[i2], :, :, :])
            xn = np.uint8(x / x.max() * 255)
            img = Image.fromarray(xn, mode='RGB')
            ax1.imshow(img)
        i += 1
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == '__main__':

    # download dataset from json object
    X, target = load_data_from_json(file = './data/shipsnet.json')

    # the target is a 2d array of the probability
    # that the image is a ship or isnt a ship
    # need to construct this 2d prob array
    y = keras.utils.np_utils.to_categorical(target)

    # plot image file
    plot_example_rgb(X[0], savefile = './images/example_input.png')

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

    #summarise
    keras2ascii(model)

    #fit or load pre-trained model
    model = fit_load_model(X_train_norm, y_train,
                              new_model=False,
                              picklefile='./models/custom_convnet.pickle',
                              input_model=model)

    # score model on test data
    y_pred = model.predict(X_test_norm)
    diagnostic_plots(y_pred, y_test,
                     labels_in=None,
                     diagnostic_file='./images/roc_plot.png',
                     max_fpr_tollerance=0.01)

    # show examples of false positives and negatives
    plot_example_classes(X_test_norm, y_pred, y_test,
                         N_examples=3, filename='./images/examples.png',
                         idx_custom=None, labels_custom=None)

    # just plot some of the original samples
    plot_example_classes(X_test_norm,
                         None, None,
                         idx_custom=[[1,2,3],[4,5,6],[7,8,9]],
                         labels_custom=['','',''],
                         N_examples = 3,
                         filename='./images/input_examples.png')





