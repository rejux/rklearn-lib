################
## adaline.py ##
################

# original implementation
# Sebatian Raschka, Python Machine Learning, 3rd Edition

#############
## imports ##
#############

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from rktools.monitors import ProgressBar

###############################################################################
##                                AdalineGD                                  ##
###############################################################################

class AdalineGD(BaseEstimator, ClassifierMixin):
    """
    The ADAptive LInear NEuron classifier.

    Parameters
    ----------

    * lr: float
        Learning rate (between 0.0 and 1.0)
    * n_epochs: int
        Passes over the training dataset.
    * random_state: int
        Random number generator seed for random weight initialization.

    Attributes
    -----------

    * w_: 1d-array
        Weights after fitting.
    * cost_: list
        Sum-of-squares cost function value in each epoch. Indeed, now
        the convergence criteria is no more the error at each epoch; but
        the value of the cost function J.
    """

    #################
    ## __init__()  ##
    #################

    # TODO Pass an potential logger as paramater

    def __init__(self, lr=0.01, n_epochs=50, random_state=1):
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state

    #####################
    ## init_weights()  ##
    #####################
    
    def init_weights(self, n_features):
        """
        Initialize the weight coefficients
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + n_features)


    ############
    ## fit()  ##
    ############

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

        # check matrices
        self.X_, self.y_ = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y) 

        # The algorithm

        self.init_weights(self.X_.shape[1])
        self.cost_ = []

        progress_bar = ProgressBar(max_value = self.n_epochs, desc="AdalineGD Epoch:")
        for i in range(self.n_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)   # here, no effect
            errors = (y - output)

            # At each epoch, the coefficients are updated using
            # the whole training dataset X, instead of one sample x_i
            self.w_[1:] += self.lr * X.T.dot(errors)
            self.w_[0] += self.lr * errors.sum()

            # cost = J(W) = 1/2 * SSE
            # with SSE = sum of error^2
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

            progress_bar.update(1)
        
        # end for
        progress_bar.close()

        return self

    ##################
    ## net_input()  ##
    ##################

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    ##################
    ## activation() ##
    ##################

    def activation(self, X):
        """Compute linear activation"""
        # Please note that the "activation" method has no effect
        # in the code since it is simply an identity function. We
        # could write `output = self.net_input(X)` directly instead.
        # The purpose of the activation is more conceptual, i.e.,
        # in the case of logistic regression 
        # we could change it to a sigmoid function to implement a logistic regression classifier.
        return X

    ###############
    ## predict() ##
    ###############

    def predict(self, X):
        """Return class label after unit step"""

        # Raise an error if not fitted
        check_is_fitted(self)
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# End AdalineGD

################################################################################
##                                AdalineSGD                                  ##
################################################################################

class AdalineSGD(BaseEstimator, ClassifierMixin):
    """
    ADAptive LInear NEuron classifier with SGD.

    Parameters
    ------------
    * lr : float
        Learning rate (between 0.0 and 1.0)
    * n_epochs : int
        Passes over the training dataset.
    * shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    * random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    -----------
    * w_ : 1d-array
        Weights after fitting.
    * cost_ : list
        Sum-of-squares cost function value averaged over all training examples in each epoch.


    """

    ################
    ## __init__() ##
    ################

    def __init__(self, lr=0.01, n_epochs=10, shuffle=True, random_state=1):
        self.lr = lr
        self.n_epochs = n_epochs
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    ###########################
    ## init_weights() ##
    ###########################

    def init_weights(self, n_features):
        """
        Initialize weights to small random numbers
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + n_features)
        self.w_initialized = True

    ############
    ## fit()  ##
    ############

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object
        
        """

        # check matrices
        self.X_, self.y_ = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y) 

        # The algorithm

        self.init_weights(X.shape[1])
        self.cost_ = []
        
        progress_bar = ProgressBar(max_value = self.n_epochs, desc="AdalineSGD Epoch:")
        for _ in range(self.n_epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
            
            progress_bar.update(1)
        
        # end for
        progress_bar.close()

        return self

    ###################
    ## partial_fit() ##
    ###################

    def partial_fit(self, X, y):
        """
        Fit training data without reinitializing the weights
        """

        if not self.w_initialized:
            self.init_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    ################
    ## _shuffle() ##
    ################

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    #######################
    ## _update_weights() ##
    #######################

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.lr * xi.dot(error)
        self.w_[0] += self.lr * error
        cost = 0.5 * error**2
        return cost

    


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X


    ###############
    ## predict() ##
    ###############

    def predict(self, X):
        """
        Return class label after unit step
        """
        
        check_is_fitted(self)

        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

# End of AdalineSGD 


