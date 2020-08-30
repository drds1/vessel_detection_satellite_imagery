# Vessel Detection From Satellite Imagery

This repo builds a binary classification model that attempts 
to classify whether or not a satellite image contains a vessel.

### Data Source
The labelled [dataset](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) 
contains 4000 (80 X 80 X 3) images stored in json format 
with a few examples shown below.

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/input_examples.png =250x250)



### Model Summary
The Kaggle notebook documents the full methodology. Here I give a brief summary. 
The approach uses a [CNN](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) model built using Python's [keras](https://keras.io/) library with a softmax 
output layer to predict the probability of each outcome (vessel or no vessel). The full architecture is as follows:


_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 80, 80, 32)        896       
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 40, 40, 32)        0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 40, 40, 32)        0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 40, 40, 32)        9248      
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 20, 20, 32)        0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 20, 20, 32)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 20, 20, 32)        9248      
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 10, 10, 32)        0         
_________________________________________________________________
dropout_13 (Dropout)         (None, 10, 10, 32)        0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 10, 10, 32)        102432    
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 5, 5, 32)          0         
_________________________________________________________________
dropout_14 (Dropout)         (None, 5, 5, 32)          0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 800)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 512)               410112    
_________________________________________________________________
dropout_15 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 2)                 1026      
=================================================================
Total params: 532,962
Trainable params: 532,962
Non-trainable params: 0
_________________________________________________________________




### Performance

The model is split 70 / 30 into build and holdout samples. The trained model
is scored on the holdout data with an expected sensitivity 
of 93% assuming a max tolerable false positive rate of 1%.

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/roc_plot.png)
