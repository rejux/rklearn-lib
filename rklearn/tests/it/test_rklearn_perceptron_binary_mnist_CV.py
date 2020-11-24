#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################
## test_rklearn_perceptron_binary_mnist_CV.py  ##
#################################################

"""
This program is an application of the Perceptron algorithm to the MNIST dataset.
Since the Perceptron is only capable of doing binary classification, we'll just use it to 
identify one digit 5. Thus, it will predict if a digit is 5 (Y) or not 5 (N).

All the data viz part is removed from this program. The interested user should confer to the 
first program of the series for Data Viz details.

Usage:
$ cd <top_dir>
$ python rklearn/tests/test_rklearn_perceptron_binary_mnist_CV.py --conf=rklearn/tests/config/config-mnist.yaml

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

from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import accuracy_score

from rktools.loggers import init_logger

from rklearn.perceptron import Perceptron
from rklearn.open_data_loaders import load_mnist_sklearn

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
        logger.info("=============================")
        logger.info("Data preparation")
        logger.info("=============================")
        logger.info("")

        # simple normalization (e.g. scale values between [0,1] 
        X_prep = X_train/255
        y_prep = y_train
      

        # just a part of the data to speed the training
        # since the data is shuffled, we can safely slice it
        # X_prep = X_prep[:1000]
        # y_prep = y_prep[:1000]

        logger.info("")
        logger.info("==================================================")
        logger.info("Cross-validate the rklearn perceptron")
        logger.info("==================================================")
        logger.info("")

        # hyperparams
        lr = config["perceptron_hyper"]["lr"]
        n_epochs = config["perceptron_hyper"]["n_epochs"]
        k = config["perceptron_hyper"]["k_fold_cv"]

        logger.info("Hyperparameters:")
        logger.info("\tlr = {}".format(lr))
        logger.info("\tnb epochs = {}".format(n_epochs))
        logger.info("\tk (-folds CV) = {}".format(k))

        logger.info("")
        start_cv = time.time()
        ppn = Perceptron(lr=lr, n_epochs=n_epochs)
        cv_scores = cross_val_score(ppn, X_prep, y_prep, cv=k, scoring="accuracy")
        avg_score = np.mean(cv_scores)
        end_cv = time.time()
        logger.info("")
        logger.info("K-fold CV duration = {} secs.".format(end_cv - start_cv))
        
        logger.info("")
        logger.info("===========================")
        logger.info("Training CV Accuracy")
        logger.info("===========================")
        logger.info("")
        logger.info("Cross-val Classification Accuracy, CV = {}: {}".format(k, cv_scores))
        logger.info("=> Mean accuracy = {:.3f} ({:.2f}%) ".format(avg_score, (avg_score * 100)))

        logger.info("")
        logger.info("===========================")
        logger.info("Testing Accuracy")
        logger.info("===========================")
        logger.info("")

        logger.info("Fitting the model...")
        logger.info("")
        # fit before (indeed, CV has trained a clone of ppn, not ppn)
        start_fit = time.time()
        ppn.fit(X_prep, y_prep)
        end_fit = time.time()
        logger.info("")
        logger.info("Done done in {} seconds".format(end_fit - start_fit))
        logger.info("")

        y_pred = ppn.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        logger.info("Misclassified examples: {} on {} samples ".format((y_test != y_pred).sum(), len(y_test)))
        logger.info("Classification Accuracy: {:.3f} ({:.2f}%) ".format(acc_score, (acc_score * 100)))

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

