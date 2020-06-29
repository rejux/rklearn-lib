#################
## plotters.py ##
#################

#############
## Imports ##
#############

# import numpy as np
# import matplotlib

from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve 

###################################
## mnist_digit_pretty_printer()  ##
###################################

def mnist_digit_pretty_printer(digit, white=" ", gray="+"):
    """
    Parameters:
    -----------
    digit: a 2D array of floats
        The pixels intensity of the digit
    white: char
        xxxx
    gray: char
        xxxx

    """

    text = ""
    for l in range(0, digit.shape[0]):
        line = ""
        for c in range(0, digit.shape[1]):
            if digit[l][c] < (Decimal(10) ** -3):
                line += white
            else:
                line += gray
        text += line + "\n"

    return text



###########################
## plot_2D_binary_data() ##
###########################

# def plot_2D_binary_data(x1, y1, x2, y2, label1 = "setosa", label2 = "versicolor"):
def plot_2D_binary_data(x1, y1, x2, y2, label1 = "class1", label2 = "class2"):
    """
    Plot the 2 classes in the 2D feature subspace: 1 feature per dimension.
    Useful to visually check that classes are lineary seperable.

    Parameters
    ----------

    TODO

    Returns
    -------
    plt: A matplotlib.plt object

    Usage
    -----
    ex. plot_binary_iris_data(X[:50, 0], X[:50, 1], X[50:100,0], X[50:100,1])

    """

    plt.close()
    plt.scatter(x1, y1, color='red', marker='o', label=label1)
    plt.scatter(x2, y2, color='blue', marker='x', label=label2)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    return plt


############################
## plot_simple_sequence() ##
############################

def plot_simple_sequence(y_values, xlabel="xlabel", ylabel="ylabel", title="No title"):
    """
    Plot a simple sequence of values: errors (e.g. y) vs epochs (e.g. x), costs vs epochs...
    The x values is deduced from the size of the y values.

    Parameters
    ----------
    y_values: list like
        The values of interest

    Returns
    -------
    A plot object
    """

    plt.close()
    plt.plot(range(1, len(y_values) + 1), y_values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return plt

#####################################
## plot_learning_curves_cv_score() ##
#####################################

# tribute: A. Géron / Hands-On Machine Learning With Scikit-Learn and Tensorflow
# tribute: Scikit-learn https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html 

def plot_learning_curves_cv_scores(estimator, 
                         X, y,  
                         title = "Learning curves", 
                         ylim = None, 
                         cv = None,
                         n_jobs = 4,
                        fit_times_vs_data = True,
                        scores_vs_fit_times = True,
                         logger = None):
    """
    Plot the learning curves of a given estimator using the Scikit Learn accuracy_score metric.

    Parameters:
        estimator: xxxx
            a estimator compliant to the Scikit Learn BaseEstimator interface (e.g. fit, transform...)
        X, y: features and labels matrices
            xxxx
        cv: xxxxx
            xxxxx

        n_jobs: int or None, optional (default=None)
            Number of jobs to run in parallel. 'None' means 1 unless in a :obj:`joblib.parallel_backend` context.
            '-1' means using all processors. See :term:`Glossary <n_jobs>` for more details. 

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        fit_times_vs_data: bool = True,
            Provide the fit_times = f(training data) curve.

        scores_vs_fit_times: bool = True,
            Provide the scores = f(fit_times) curve.
 

    Returns:
        plt: the plot object

    """

    # using the Scikit-learn API
    # train_sizes = [int(s) for s in np.linspace(1, X.shape[0], X.shape[0])] # absolute sizes
    train_sizes = np.linspace(.1, 1.0, 10) # N = 10 relative sizes or ratios 
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)

    # metrics 

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # 3 Plots:
    # - lerning curves or scores = f(n_samples),
    # - fit_times = f(n_samples)
    # - score = f(fit_times)

    if fit_times_vs_data is True:
        if scores_vs_fit_times is True:
            _, axes = plt.subplots(3, 1, figsize=(8, 20))
        else: 
            _, axes = plt.subplots(2, 1, figsize=(8, 15))
    else:
        if scores_vs_fit_times is True: 
            _, axes = plt.subplots(2, 1, figsize=(8, 15))
        else: 
            _, axes = plt.subplots(1, 1, figsize=(8, 8))

    #
    # Plot learning curve
    #

    if ylim is not None:
        axes[0].set_ylim(*ylim)

    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title(title)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    #
    # Plot n_samples vs fit_times = f(n_samples)
    #

    if fit_times_vs_data is True:
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # index of the next plot: scores = f(fit_times) 
            
        if scores_vs_fit_times is True:
            index = 2
        else: # => N/A
            index = -1 
    else: 
        index = 1 

    if index > 0:

        axes[index].grid()
        axes[index].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[index].fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
        axes[index].set_xlabel("fit_times")
        axes[index].set_ylabel("Score")
        axes[index].set_title("Performance of the model")
    
    return plt


####################################
## plot_learning_curves_cv_rmse() ##
####################################

    # Custom code

#    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_ratio)
#
#    if logger:
#        msg = "[plot_learning_curves_acc_score()] X_train.shape = {}, y_train.shape = {}, X_val.shape = {}, y_val.shape = {}"
#        logger.debug(msg.format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
#
#     train_accs, val_accs = [], []
#     for m in range(1, len(X_train)):
# 
#         estimator.fit(X_train[:m], y_train[:m])
#         y_train_predict = estimator.predict(X_train[:m])
#         y_val_predict = estimator.predict(X_val)
# 
#         # calculate the errors
#         training_acc = accuracy_score(y_train_predict, y_train[:m])
#         validation_acc = accuracy_score(y_val_predict, y_val)
# 
#         train_accs.append(training_acc)
#         val_accs.append(validation_acc)
# 
#     plt.close()
#     plt.plot(train_accs, "r-+", linewidth=2, label="train")
#     plt.plot(val_accs, "b-", linewidth=3, label="val")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend(loc="best")
#     return plt








