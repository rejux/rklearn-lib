#################################
## test_rklearn_perceptron.py  ##
#################################

# Usage:
# (...) $ cd <top_dir>
# (...) $ python rklearn/tests/test_rklearn_perceptron.py --conf rklearn/tests/config/config.yaml

#############
## Imports ##
#############

import unittest

import os
import sys
import argparse
import yaml
import numpy as np

from rktools.loggers import init_logger
from rklearn.perceptron import Perceptron

#############
## Globals ##
#############

###########################
## TestRkLearnPerceptron ##
###########################

class TestRkLearnPerceptron(unittest.TestCase):

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

        # fixture: the data X,y
        self.X = np.array(
                [[5.1,3.5,1.4,0.2],
             [4.9,3.0,1.4,0.2],
             [4.7,3.2,1.3,0.2],
             [4.6,3.1,1.5,0.2],
             [5.0,3.6,1.4,0.2],
             [5.4,3.9,1.7,0.4],
             [4.6,3.4,1.4,0.3],
             [5.0,3.4,1.5,0.2],
             [7.0,3.2,4.7,1.4],
             [6.4,3.2,4.5,1.5],
             [6.9,3.1,4.9,1.5],
             [5.5,2.3,4.0,1.3],
             [6.5,2.8,4.6,1.5],
             [5.7,2.8,4.5,1.3],
             [6.3,3.3,4.7,1.6],
             [4.9,2.4,3.3,1.0]])

        self.y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1])

        logger.info("self.X = {}, self.X.shape = {}".format(self.X, self.X.shape))
        logger.info("self.y = {}, self.y.shape = {}".format(self.y, self.y.shape))

        # the fixture
        self.eta = 0.01
        self.n_epochs = 50
        self.perceptron = Perceptron(self.eta, self.n_epochs)

        logger.info("self.perceptron = {}".format(self.perceptron))

    #######################
    ## test_net_input()  ##
    #######################

    def test_net_input(self):
        
        assert(logger is not None)
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("######################")
        logger.info("## test_net_input() ##")
        logger.info("######################")
        logger.info("")

        self.perceptron.init_weights(self.X.shape[1])
        logger.info("self.perceptron.net_input(X) = {}".format(self.perceptron.net_input(self.X)))

        logger.info("End of test_net_input()")

    def test_net_input_2(self):
        
        assert(logger is not None)
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("########################")
        logger.info("## test_net_input_2() ##")
        logger.info("########################")
        logger.info("")

        self.perceptron.init_weights(self.X.shape[1])
        logger.info("self.perceptron.net_input(X[0]) = {}".format(self.perceptron.net_input(self.X[0])))

        logger.info("End of test_net_input_2()")

    def test_net_input_3(self):
        
        assert(logger is not None)
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("########################")
        logger.info("## test_net_input_3() ##")
        logger.info("########################")
        logger.info("")

        self.perceptron.w_ = np.array([-0.00375655, -0.02811756, -0.07728172, 0.09327031, 0.05265408])
        # self.perceptron.init_weights(self.X.shape[1])
        logger.info("self.perceptron.w_ = {}".format(self.perceptron.w_))
        logger.info("self.perceptron.net_input(X[0]) = {}".format(self.perceptron.net_input(self.X[0])))

        logger.info("End of test_net_input_2()")


    ######################
    ## test_predict_1() ##
    ######################

    def test_predict_1(self):

        assert(logger is not None)
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("######################")
        logger.info("## test_predict_1() ##")
        logger.info("######################")
        logger.info("")

        self.perceptron.w_ = np.array([-0.00375655, -0.02811756, -0.07728172, 0.09327031, 0.05265408])
        # self.perceptron.w_ = np.array([0.002, 0.1, -0.2, 0.3, 0.02])    # w_[0] is the bias
        logger.info("self.perceptron.w_ = {}".format(self.perceptron.w_))

        for xi in self.X:
            logger.info("\txi = {}, self.perceptron.predict(xi) = {}".format(xi, self.perceptron.predict(xi)))

        logger.info("End of test_predict_1()")
     
    
    ########################
    ## test_fit_predict() ##
    ########################

    def test_fit_predict(self):

        assert(logger is not None)
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("########################")
        logger.info("## test_fit_predict() ##")
        logger.info("########################")
        logger.info("")

        logger.info("Fit the perceptron...")
        self.perceptron.fit(self.X, self.y)

        logger.info("\tself.perceptron.w_ = {}".format(self.perceptron.w_))

        logger.info("")
        logger.info("Predict the training data...")
        for xi in self.X:
            logger.info("\txi = {}, self.perceptron.predict(xi) = {}".format(xi, self.perceptron.predict(xi)))

        logger.info("End of test_predict()")


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

        self.perceptron = None


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
    flags = None
    config = None

    # cmd line arguments
    flags, argv = parse_args()
    sys.argv[:] = argv

    with open(flags.conf, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    logger = init_logger(name="test_rklearn_perceptron", config = config)

    logger.info("")
    logger.info("################################")
    logger.info("## test_rklearn_perceptron.py ##")
    logger.info("################################")
    logger.info("")

    unittest.main()


