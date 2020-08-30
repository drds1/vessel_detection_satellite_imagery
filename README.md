# Vessel Detection From Satellite Imagery

This repo builds a binary classification model that attempts 
to classify whether or not a satellite image contains a vessel.

### Data Source
The labelled [dataset](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) 
contains 4000 (80 X 80 X 3) images stored in json format 
with a few examples shown below.

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/images/input_examples.png)


 
### Model Summary - See also kaggle notebook

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

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/images/roc_plot.png)



### Misclassifications

By examining the false positive / negative classifications, 
we can try to understand the features in images that might lead the model to
misclassify a particular image. A few of these are plotted below.

![](https://github.com/dstarkey23/transfer_learning_computer_vision/blob/master/images/examples.png)

We see the model correctly identifies when we are looking at empty see or some other landmass
(i.e. an empty birth or building or road). This is good. Unfortunately the model appears to interpret
ship-to-ship moorings (when two vessels dock with each other) as an alien structure and cannot
discern the docked vessels as individual ships. Other misclassifications occur when a vessel is
half in the frame and half out. It seems from the far-right true negative panel that the labels
are defined to exclude ships partially in-frame. Ships that are almost fully in-frame 
(as shown in the right false positive panel) are considered as vessels by the model but labelled as 'not vessels'
during training. Edge cases such as these, where the label is a little muddled, will naturally confuse the CNN
and lead to such misclassifications. Despite these shortcomings, we see that the model on the hole identifies vessels 
in satellite images with exceptional performance. 

Thanks for reading! Please feel free to get in touch and suggest tweaks to the approach
or areas for performance uplift.