#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################
## test_rklearn_tfoo_v1_cifar_10_cnn.py ##
##########################################

"""
This program is an implementation/test of binary classifier written with Tensorflow v1.x.
It uses our one module framework tfoo_v1. 

The objective here is essentially to port a classical TF 1.x code to an OO one based on our 
one module fwk. For this we'll reuse the odel defined by XXX in XXX. 




Usage:
$ cd <top_dir>
$ python rklearn/tests/test_rklearn_tfoo_v1_cifar_10_cnn.py --conf rklearn/tests/config/config.yaml
"""

#############
## Imports ##
#############

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import os
import sys
import time
import argparse
import yaml
import tensorflow as tf

from bunch import Bunch 

# Objects
from cifar10_data_generator import CIFAR10DataGenerator 
from cifar10_cnn import CIFAR10CNN

from rklearn.tfoo_v1 import BaseTrainer, BaseGraph

from rktools.loggers import init_logger



####################
## CIFAR10Trainer ##
####################

class CIFAR10Trainer(BaseTrainer):

    def __init__(self, sess, model, data_gen, config, tb_logger = None, logger = None):
        super().__init__(sess, model, data_gen, config, tb_logger, logger)


####################################################################################################
##                                               main()                                           ##
####################################################################################################

def main(config, logger = None):

    try:
        
        logger.info("\n# 1. Loading the CIFAR10 data...")
        data_gen = CIFAR10DataGenerator(config, logger) 
        data_gen.load_data()
        logger.info("(before preparation) data_gen.data_shapes() = {}".format(data_gen.data_shapes()))
        logger.info("(before preparation) data_gen.X_train[0] = {}, data_gen.y_train[0] = {}".format(data_gen.X_train[0], data_gen.y_train[0]))
       
        logger.info("\n# 2. Preprocessing data...")
        data_gen.prepare_data()
        logger.info("(after preparation) data_gen.data_shapes() = {}".format(data_gen.data_shapes()))

        logger.info("\n# 3. Graph construction...") 
        BaseGraph.reset_graph()
        model = CIFAR10CNN(config, logger)
        model.build_model()
        model.init_checkpoint_saver()

        logger.info("\n# 4. Graph execution...") 

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config = config) as sess:
            trainer = BaseTrainer(sess = sess, model = model, data_gen = data_gen, config = config, logger = logger)
            trainer.var_init.run()
            trainer.train()
            
            # Determine success rate
            logger.info(">>>> Determine the success rate on test data:")
            logger.info("\tAccuracy = {}".format(trainer.eval_performance()))


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

    FLAGS, argv = parse_args()
    sys.argv[:] = argv

    with open(FLAGS.conf, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    assert(config is not None)

    # bunch it to facile usage 
    config = Bunch(config)

    logger = init_logger(name="test_rklearn_tfoo_v1_cifar_10_cnn", config = config)


    logger.info("")
    logger.info("##########################################")
    logger.info("## test_rklearn_tfoo_v1_cifar_10_cnn.py ##")
    logger.info("##########################################")
    logger.info("")

    main(config, logger = logger)


