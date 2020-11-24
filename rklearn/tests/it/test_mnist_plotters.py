#############################
## test_mnist_plotters.py  ##
#############################

# WARNING:
# Run the script extract_mnist_samples.py to generate data for these tests.
# The data will be stored in config[mnist_data][samples_dir]


# Usage:
# (...) $ cd <top_dir>
# (...) $ python rklearn/tests/test_mnist_plotters.py --conf=rklearn/tests/config/config-mnist.yaml

#############
## Imports ##
#############

import unittest

import os
import sys
# import time
import logging
import argparse
import yaml
import numpy as np

from rklearn.plotters import mnist_digit_pretty_printer
from rktools.loggers import init_logger

#############
## Globals ##
#############

logger = None
flags = None
config = None

#######################
## TestMNISTPlotters ##
#######################

class TestMNISTPlotters(unittest.TestCase):

    ##############
    ## setUp()  ##
    ##############

    def setUp(self):
        """
        setUp
        """
        assert(logger is not None)

        logger.info("")
        logger.info("##############")
        logger.info("## setUp()  ##")
        logger.info("##############")
        logger.info("")

    #######################
    ## test_net_input()  ##
    #######################

    def test_net_input(self):
        """
        test
        """
        assert(logger is not None)
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("######################")
        logger.info("## test_net_input() ##")
        logger.info("######################")
        logger.info("")


        logger.info("End of test_net_input()")


    #################
    ## tearDown()  ##
    #################

    def tearDown(self):

        assert(logger is not None)

        logger.info("")
        logger.info("#################")
        logger.info("## tearDown()  ##")
        logger.info("#################")
        logger.info("")



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

    flags, argv = parse_args()
    # print(flags, argv)
    sys.argv[:] = argv

    with open(flags.conf, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger = init_logger(name="test_mnist_plotters",
            level=logging.getLevelName(config["logger"]["log_level"].upper()))

    logger.info("")
    logger.info("############################")
    logger.info("## test_mnist_plotters.py ##")
    logger.info("################################")
    logger.info("")

    unittest.main()


