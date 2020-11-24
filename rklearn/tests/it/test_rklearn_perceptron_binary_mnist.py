#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################
## test_rklearn_perceptron_binary_mnist.py  ##
##############################################

"""
This program is an application of the Perceptron algorithm to the MNIST dataset.
Since the Perceptron is only capable of doing binary classification, we'll just use it to 
identify one digit 5. Thus, it will predict if a digit is 5 (Y) or not 5 (N).

Usage:
$ cd <top_dir>
$ python rklearn/tests/test_rklearn_perceptron_binary_mnist.py --conf=rklearn/tests/config/config-mnist.yaml

"""



#############
## Imports ##
#############

import unittest

import os
import sys
import time
import argparse
import yaml
import numpy as np ; np.random.seed(1) # ; np.set_printoptions(threshold=np.inf)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from rktools.loggers import init_logger

from rklearn.perceptron import Perceptron
from rklearn.open_data_loaders import load_mnist_sklearn
from rklearn.plotters import mnist_digit_pretty_printer, plot_simple_sequence

#############################
## binarize_mnist_labels() ##
#############################

def binarize_mnist_labels(y, pos_class):
    """
    Create the 2 classes in the whole set: pos_class set to 1 and others set to 0
    """
    return np.where(y == pos_class, 1, 0) # pos_class => 1, all others => 0



#############
## main()  ##
#############

def main(config):

    try:

        start = time.time()

        logger.info("")
        logger.info("===========================")
        logger.info("Load the MNIST data...")
        logger.info("===========================")
        logger.info("")

        os.makedirs(config["mnist_data"]["data_home"], exist_ok=True)
        
        start_prep = time.time()
        X,y = load_mnist_sklearn(data_home=config["mnist_data"]["data_home"], shuffle = True, logger = logger)
        end_prep = time.time()
        logger.info("Data loaded in {} seconds".format(end_prep - start_prep))

        logger.info("X.shape = {}, y.shape = {}".format(X.shape, y.shape))
        logger.debug("X.sample = \n{}, \ny.sample = {}".format(X[:5,], y[:5]))


        logger.info("")
        logger.info("===========================")
        logger.info("Labels binarization...")
        logger.info("===========================")
        logger.info("")
        
        # get the index of the first element = pos_class sample
        pos_class = config["mnist_data"]["pos_class"]
        logger.info("Positive class = {} for this use case. All over are negatives.".format(pos_class))
 
        # get the indices of one sample of pos_class, and another of a negative class

        try:
            index_pos_sample = y.tolist().index(pos_class)
        except:
            raise Exception("No element equal to {} (e.g. pos_class) was found in the target vector. Abort!".format(pos_class))

        # we can safely assume that the index before is a neg class

        if index_pos_sample > 0:
            index_neg_sample = index_pos_sample - 1 
        else: 
            index_neg_sample = index_pos_sample + 1

        logger.info("index_pos_sample = {} => label = {}".format(index_pos_sample, y[index_pos_sample]))
        logger.info("index_neg_sample = {} => label = {}".format(index_neg_sample, y[index_neg_sample]))

        # Labels binarization
        y_bin = binarize_mnist_labels(y, pos_class)

        logger.info("(after binarization) Labels samples, y_bin[:10] = {}".format(y_bin[:10]))

        logger.info("")
        logger.info("=============================")
        logger.info("Train-test split the data")
        logger.info("=============================")
        logger.info("")

        # train: 0 -> 60K - 1 / test: 60K -> 70K => ratio test vs train = 15% 
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, train_size=60000, random_state=1, stratify=y_bin)

        logger.info("X_train.shape = {}, y_train.shape = {}".format(X_train.shape, y_train.shape))
        logger.info("X_test.shape = {}, y_test.shape = {}".format(X_test.shape, y_test.shape))

        logger.info("")
        logger.info("Check the stratifications in both train and test sets:")
        
        bin_count = np.bincount(y_bin)
        logger.info("Labels counts in y_bin = {} => percentage neg/pos = {:.2f}%".format(bin_count, ((bin_count[1]/bin_count[0])*100)))

        bin_count = np.bincount(y_train)
        logger.info("Labels counts in y_train = {} => percentage neg/pos = {:.2f}%".format(bin_count, ((bin_count[1]/bin_count[0])*100)))

        bin_count = np.bincount(y_test)
        logger.info("Labels counts in y_test = {} => percentage neg/pos = {:.2f}%".format(bin_count, ((bin_count[1]/bin_count[0])*100)))

        logger.info("")
        logger.info("=========================================")
        logger.info("Data viz: display some samples data")
        logger.info("=========================================")
        logger.info("")

        # recalculate the index_pos_sample, and index_neg_sample, because 
        # split hasmodified the indexing

        try:
            index_pos_sample = y_train.tolist().index(1)  # pos_class is now 1
        except:
            raise Exception("No element equal to 1 (e.g. pos_class) was found in the target vector. Abort!")

        if index_pos_sample > 0:
            index_neg_sample = index_pos_sample - 1 
        else: 
            index_neg_sample = index_pos_sample + 1

        logger.info("A pos_class sample at index {}: ".format(index_pos_sample))
        logger.info("new label = {}".format(y_train[index_pos_sample]))
        logger.info("raw pixels values = \n{}".format(np.reshape(X_train[index_pos_sample], (-1, 28))))
        logger.info("pretty print version = \n{}".format(mnist_digit_pretty_printer(np.reshape(X_train[index_pos_sample], (-1, 28)))))
        
        logger.info("A neg_class sample ayt index {}: ".format(index_neg_sample))
        logger.info("new label = {}".format(y_train[index_neg_sample]))
        logger.info("raw pixels values = \n{}".format(np.reshape(X_train[index_neg_sample], (-1, 28))))
        logger.info("pretty print version = \n{}".format(mnist_digit_pretty_printer(np.reshape(X_train[index_neg_sample], (-1, 28)))))

        logger.info("")
        logger.info("=============================")
        logger.info("Data preparation")
        logger.info("=============================")
        logger.info("")

        # simple normalization (e.g. scale values between [0,1] 
        X_prep = X_train/255
        y_prep = y_train

        logger.info("Some data viz to check quickly the (X_prep, y_prep) matrices...")
        logger.info("\tX_prep.shape = {}, y_prep.shape = {}".format(X_prep.shape, y_prep.shape))
        logger.info("")
        logger.info("The pos_class sample at index {} (after data prep phase): ".format(index_pos_sample))
        logger.info("\tnew label = {}".format(y_prep[index_pos_sample]))
        logger.info("\traw pixels values = \n{}".format(np.reshape(X_prep[index_pos_sample], (-1, 28))))
        logger.info("\tpretty print version = \n{}".format(mnist_digit_pretty_printer(np.reshape(X_prep[index_pos_sample], (-1, 28)))))

        logger.info("")
        logger.info("===================================")
        logger.info("Fitting the rklearn perceptron")
        logger.info("===================================")
        logger.info("")

        # training plot
        os.makedirs(config["mnist_data"]["plots_dir"], exist_ok=True)
        train_error_fig = config["perceptron_hyper"]["train_error_fig"]

        # hyperparams
        lr = config["perceptron_hyper"]["lr"]
        n_epochs = config["perceptron_hyper"]["n_epochs"]

        logger.info("Hyperparameters:")
        logger.info("\tlr = {}".format(lr))
        logger.info("\tnb epochs = {}".format(n_epochs))

        start_fit = time.time()
        ppn = Perceptron(lr=lr, n_epochs=n_epochs)
        ppn.fit(X_prep, y_prep)
        end_fit = time.time()
        logger.info("Fit done in {} seconds".format(end_fit - start_fit))
        logger.info("Training errors for all epochs = {}".format(ppn.errors_))

        suffix = "perceptron-lr-{}-epochs-{}".format(lr, n_epochs)
        fig = train_error_fig.format(suffix)
        plot_simple_sequence(ppn.errors_,
                xlabel="Epochs", ylabel="Errors",
                title="Training errors = f(Epochs) - lr = {}".format(lr)).savefig(fig, dpi=300)
        logger.info("Plotted trainig errors in {}".format(fig))

        logger.info("")
        logger.info("===========================")
        logger.info("Testing Accuracy")
        logger.info("===========================")
        logger.info("")

        # Simple accuracy
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

    logger = None
    FLAGS = None
    config = None

    # Parse cmd line arguments
    FLAGS, argv = parse_args()
    sys.argv[:] = argv

    with open(FLAGS.conf, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    assert(config is not None)

    logger = init_logger(name="test_mnist_perceptron", config = config)

    logger.info("")
    logger.info("#############################################")
    logger.info("## test_rklearn_perceptron_binary_mnist.py ##")
    logger.info("#############################################")
    logger.info("")

    main(config)

