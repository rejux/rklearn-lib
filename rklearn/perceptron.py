###################
## perceptron.py ##
###################

## original implementation: Sebatian Raschka, Python Machine Learning, 3rd Edition

## Here, we make it compatible with Scikit Learn tools by being compliant to the Estmator interface
## See https://scikit-learn.org/stable/developers/develop.html

##
## imports
##

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from rktools.monitors import ProgressBar

################
## Perceptron ##
################


class Perceptron(BaseEstimator, ClassifierMixin):

    """
    Perceptron classifier.

    Parameters
    -----------

    * lr : float
        Learning rate (between 0.0 and 1.0)
    * n_epochs : int
        Passes over the training dataset.
    * random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    -----------
    * w_ : 1d-array
        Weights after fitting.
    * errors_ : list
        Number of misclassifications (updates) in each epoch.

    Note: Attributes are importants fro Scikit Learn compatibility. See check_if_fitted() doc.

    """

    ##############################
    ## __init__()
    ##############################

    def __init__(self, lr=0.01, n_epochs=50, random_state=1, ascii = False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.ascii = ascii 

    ##############################
    ## init_weights()
    ##############################

    def init_weights(self, n_features):
        """
        Initialize the weight coefficients
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + n_features)

    ##############################
    ## net_input()
    ##############################

    def net_input(self, X):
        """Calculate net input"""

        return np.dot(X, self.w_[1:]) + self.w_[0]

    ##############################
    ## fit()
    ##############################

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        * X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.
        * y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        * self : object
        """

        # Check that X and y have correct shape
        # and dump them as attribute
        self.X_, self.y_ = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y) 

        # The algorithm

        self.init_weights(self.X_.shape[1])
        self.errors_ = []

        progress_bar = ProgressBar(max_value = self.n_epochs, desc="Epoch: ", ascii = self.ascii)
        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(self.X_, self.y_):
                update = self.lr * (yi - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update * 1

                # collect the number of errors ate each epoch
                errors += int(update != 0.0)

            self.errors_.append(errors)

            progress_bar.update(1)

        # end for

        progress_bar.close()

        return self

    ##############################
    ## predict()
    ##############################

    def predict(self, X):
        """Return class label after unit step"""

        # Check is fit had been called
        # the presence of self.w_, and self.errors_ suffice
        check_is_fitted(self)

        # FIXME Input validation
        # It modifies the type of X which makes it not compatible with 
        # the net_input() function
        # X = check_array(X)

        return np.where(self.net_input(X) >= 0.0, 1, -1)


# End Perceptron
