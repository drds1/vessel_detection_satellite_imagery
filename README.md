# Vessel Detection From Satellite Imagery

This repo builds a binary classification model that attempts 
to classify whether or not a satellite image contains a vessel.

### Data Source
The labelled [dataset](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) 
contains 4000 (80 X 80 X 3) images stored in json format 
with a few examples shown below.

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/input_examples.png)



### Model Summary
The Kaggle notebook documents the full methodology. Here I give a brief summary. 
The approach uses a [CNN](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) model built using Python's [keras](https://keras.io/) library with a softmax 
output layer to predict the probability of each outcome (vessel or no vessel). The full architecture is as follows:

           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

               Input   #####     80   80    3
              Conv2D    \|/  -------------------       896     1.6%
                relu   #####     80   80   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     40   40   32
             Dropout    | || -------------------         0     0.0%
                       #####     40   40   32
              Conv2D    \|/  -------------------      9248    17.0%
                relu   #####     40   40   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     20   20   32
             Dropout    | || -------------------         0     0.0%
                       #####     20   20   32
              Conv2D    \|/  -------------------      9248    17.0%
                relu   #####     20   20   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####     10   10   32
             Dropout    | || -------------------         0     0.0%
                       #####     10   10   32
              Conv2D    \|/  -------------------      9248    17.0%
                relu   #####     10   10   32
        MaxPooling2D   Y max -------------------         0     0.0%
                       #####      5    5   32
             Dropout    | || -------------------         0     0.0%
                       #####      5    5   32
             Flatten   ||||| -------------------         0     0.0%
                       #####         800
               Dense   XXXXX -------------------     25632    47.2%
                relu   #####          32
             Dropout    | || -------------------         0     0.0%
                       #####          32
               Dense   XXXXX -------------------        66     0.1%
             softmax   #####           2



### Performance

The model is split 70 / 30 into build and holdout samples. The trained model
is scored on the holdout data with an expected sensitivity 
of 93% assuming a max tolerable false positive rate of 1%.

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/roc_plot.png)
