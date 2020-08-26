#image analysis project from vtx
#add training dataset from kaggle project
#https://www.kaggle.com/rhammell/ships-in-satellite-imagery
#paper https://www.irjet.net/archives/V6/i9/IRJET-V6I9291.pdf

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import glob


if __name__ == '__main__':
    data_dir = './data/images/shipsnet/shipsnet/'
    #list all files image files in dataset
    files = glob.glob(data_dir+'*.png')
    files = [f.replace(data_dir,'') for f in files]

    #the target is given in the image file name by the prefix 1 or 0
    #extract this information
    target = np.array([int(f[0]) for f in files])
    print('target info extracted')
    for t in np.unique(target):
        print('found '+str(len(target[target==t]))+'images with class label '+str(t))
