######################
## data_loaders.py  ##
######################

#############
## Imports ##
#############

import os
import sys
# import yaml
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml



##########################
## load_mnist_sklearn() ##
##########################

def load_mnist_sklearn(data_home = None, shuffle = False, logger = None):
    """
    Download the MNIST data from openml.org using the Scikit Learn API.
    Confer to  for more details. 
        - 70K images
        - 1 image has 784 features (e.g. 28x28 pixels), and corresponds 
        to 1 label (e.g. one digit in {0, 1, ..., 9})
        - 1 feature value represents the pixel inetsity from 0 (black) to 255 (white). 
        Thus each image is a grayscale one  

    Parameters:
    -----------
    data_home: string
        xxxx
    shuffle: boolean
        xxxx
    logger: object
        xxxx

    Returns:
    --------
    X, y: The features and labels matrices

    """

    # Load data from https://www.openml.org/d/554
    if os.path.exists(data_home): 
        
        if logger:
            logger.debug("Loading data into {}".format(data_home))

        X,y = fetch_openml('mnist_784', data_home=data_home, version=1, return_X_y=True)
    
    else:

        if logger:
            logger.debug("Loading data into default location = ~/sklearn_data")

        X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

    # convert the y, it seems to be array of chars
    y = y.astype(int)

    if shuffle:

        if logger:
            logger.debug("Shuffling data...")

        shuffle_index = np.random.permutation(len(X))
        X = X[shuffle_index]
        y = y[shuffle_index]
        
    return X,y      

#############################
## load_iris_binary_data() ##
#############################

def load_iris_binary_data(
        csv_file,
        features = [0,2],
        pos_class = "Iris-setosa",
        neg_class = "Iris-versicolor",
        logger = None):

    """
    Prepare the data for both training a binary classifier for setosa and versicolor

    Parameters
    ----------
    csv_file: string
        The absolute path of the CSV file that contains data (both training and testing)

    features = list like, default = [0,2]
        The retained columns (e.g. sepal-length=0, petal-length=2 )

    Returns
    -------
    X,y: The features and labels matrices that correspond to the 2 classes 
    
    """
    try:

        if logger:
            logger.debug("Loading IRIS data from file {}:".format(csv_file))

        df = pd.read_csv(csv_file, header=None)

        if logger:
            logger.debug("Data successfully loaded. Samples: \n{}".format(df.tail()))

        # Training matrix

        y = df.iloc[0:100, 4].values        # the 2 classes corresponds to the 100 first lines
        y = np.where(y == pos_class, 1, 0)  # positve class => 1, negative => 0

        # extract sepal length and petal length
        X = df.iloc[0:100, features].values

        return X,y

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        if logger:
            logger.error("error msg = {}, error type = {}, error file = {}, error line = {}".format(e, exc_type, fname, exc_tb.tb_lineno))
        return None,None



