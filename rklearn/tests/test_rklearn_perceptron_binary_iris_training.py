######################################################
## test_rklearn_perceptron_binary_iris_training.py  ##
######################################################

"""
Test the Perceptron classifier on the iris dataset.
Since thePerceptron is a binary classifier, then only 2 categories of flowers are selected: Setosa, and Versicolor.

Usage:
$ cd <top_dir>
$ python rklearn/tests/test_rklearn_perceptron_binary_iris_training.py --conf=rklearn/tests/config/config.yaml
"""

#############
## Imports ##
#############

import unittest

import os
import sys
import time
import logging
import argparse
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from rklearn.plotters import plot_2D_binary_data, plot_simple_sequence
from rklearn.perceptron import Perceptron
from rklearn.opendata_loaders import load_iris_binary_data

#############
## Globals ##
#############

logger = None
FLAGS = None
config = None

######################
## init_logger()
######################

def init_logger(name="_NO_NAME_", level=logging.DEBUG):
    """
    Init the logger.

    Parameters
    -----------
    name: string
        The logger name
    level: value
        A value among
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            ...

    Returns
    -------
    _logger: logging.logger instance
    """

    _logger=logging.getLogger(name)
    if not len(_logger.handlers):
        _logger.setLevel(level)
        logger_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(logger_format)

        # console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        _logger.addHandler(ch)

        # if a log file is provided, then create a filehandler:
        if len(config["logger"]["log_file"]) > 0:
            fh = logging.FileHandler(config["logger"]["log_file"])
            fh.setFormatter(formatter)
            _logger.addHandler(fh)

    _logger.info("logger successfully initialized!")

    return _logger


############
## main() ##
############

# TODO Pass only the config object here

def main(
    csv_file,
    lr = 0.1,
    n_epochs = 10,
    features = [0,2],
    pos_class = "Iris-setosa",
    neg_class = "Iris-versicolor",
    data_fig = "/tmp/iris-data-fig.png",
    train_error_fig = "/tmp/iris-train-error-fig.png"):

    """
    Parameters:
    -----------

    """
    try:

        start = time.time()

        logger.info("===========================")
        logger.info("Load the data")
        logger.info("===========================")

        start_prep = time.time()
        X,y = load_iris_binary_data(csv_file = csv_file,
                features = features, pos_class = pos_class, neg_class = neg_class)
        end_prep = time.time()
        logger.info("- Data loaded in {} seconds".format(end_prep - start_prep))

        logger.debug("(after data prep) \nX.sample = \n{}, \ny.sample = {}".format(X[:5,], y[:5]))
        logger.info("(after data prep) X.shape = {}, y.shape = {}".format(X.shape, y.shape))

        # plot the data and save
        suffix = "nb-samples-{}-nb-features-{}".format(X.shape[0], X.shape[1])
        fig = data_fig.format(suffix)
        plot_2D_binary_data(X[:50, 0], X[:50, 1], X[50:100,0], X[50:100,1], label1="setosa", label2="versicolor").savefig(fig, dpi=300)

        logger.info("=============================")
        logger.info("Train-test split the data")
        logger.info("=============================")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        logger.info("X_train.shape = {}, y_train.shape = {}".format(X_train.shape, y_train.shape))
        logger.info("X_test.shape = {}, y_test.shape = {}".format(X_test.shape, y_test.shape))

        # check the stratifications in both train and test sets
        logger.info("Labels counts in y = {}".format(np.bincount(y)))
        logger.info("Labels counts in y_train = {}".format(np.bincount(y_train)))
        logger.info("Labels counts in y_test = {}".format(np.bincount(y_test)))

        logger.info("===========================")
        logger.info("Fitting a perceptron")
        logger.info("===========================")

        logger.info("Hyperparameters:")
        logger.info("- lr = {}".format(lr))
        logger.info("- nb epochs = {}".format(n_epochs))

        start_fit = time.time()
        ppn = Perceptron(lr=lr, n_epochs=n_epochs)
        ppn.fit(X_train, y_train)
        end_fit = time.time()
        logger.info("Fit done in {} seconds".format(end_fit - start_fit))
        logger.info("Training errors for all epochs = {}".format(ppn.errors_))

        suffix = "perceptron-lr-{}-epochs-{}".format(lr, n_epochs)
        fig = train_error_fig.format(suffix)
        plot_simple_sequence(ppn.errors_,
                xlabel="Epochs", ylabel="Errors",
                title="Training errors = f(Epochs) - lr = {}".format(lr)).savefig(fig, dpi=300)
        logger.info("Plotted trainig errors in {}".format(fig))

        logger.info("===========================")
        logger.info("Accuracy")
        logger.info("===========================")

        y_pred = ppn.predict(X_test)
        logger.info("Misclassified examples: {} on {} samples ".format((y_test != y_pred).sum(), len(y_test)))
        logger.info("Classification Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

        end = time.time()
        logger.info("Total duration = {} secs.".format(end - start))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("error msg = {}, error type = {}, error file = {}, error line = {}".format(e, exc_type, fname, exc_tb.tb_lineno))

###################
## parse_args()  ##
###################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="Path to the YAML configuration file", required=True,)
    ns, args = parser.parse_known_args(namespace=unittest)
    return ns, sys.argv[:1] + args

###############
## __main__  ##
###############

if __name__ == '__main__':

    FLAGS, argv = parse_args()
    sys.argv[:] = argv

    with open(FLAGS.conf, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger = init_logger(name="test_rklearn_perceptron",
            level=logging.getLevelName(config["logger"]["log_level"].upper()))

    logger.info("")
    logger.info("#####################################################")
    logger.info("## test_rklearn_perceptron_binary_iris_training.py ##")
    logger.info("#####################################################")
    logger.info("")

    main(

        # data
        csv_file = config["iris_binary_classifier"]["cvs_file"],
        features = config["iris_binary_classifier"]["features"],
        pos_class = config["iris_binary_classifier"]["pos_class"],
        neg_class = config["iris_binary_classifier"]["neg_class"],
        data_fig = config["iris_binary_classifier"]["data_fig"],

        # training plot
        train_error_fig = config["perceptron_hyper"]["train_error_fig"],

        # hyperparams
        lr = config["perceptron_hyper"]["lr"],
        n_epochs = config["perceptron_hyper"]["n_epochs"]

    )



