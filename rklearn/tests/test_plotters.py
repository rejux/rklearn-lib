#######################
## test_plotters.py  ##
#######################


# Usage:
# (...) $ cd <top_dir>
# (...) $ python rklearn/tests/test_plotters.py --conf=rklearn/tests/config/config.yaml

#############
## Imports ##
#############

import unittest

import os
import sys
import time
import argparse
import yaml

from rklearn.plotters import plot_learning_curves_cv_scores 
from rktools.loggers import init_logger
from rklearn.perceptron import Perceptron
from rklearn.opendata_loaders import load_iris_binary_data

#############
## Globals ##
#############

logger = None
flags = None
config = None

##################
## TestPlotters ##
##################

class TestPlotters(unittest.TestCase):

    ##############
    ## setUp()  ##
    ##############

    def setUp(self):
        
        assert(logger is not None)

        logger.info("")
        logger.info("##############")
        logger.info("## setUp()  ##")
        logger.info("##############")
        logger.info("")

        # config params
        self.csv_file = config["iris_binary_classifier"]["cvs_file"]
        self.features = config["iris_binary_classifier"]["features"]
        self.pos_class = config["iris_binary_classifier"]["pos_class"]
        self.neg_class = config["iris_binary_classifier"]["neg_class"]

        # hyperparams
        self.lr = config["perceptron_hyper"]["lr"]
        self.n_epochs = config["perceptron_hyper"]["n_epochs"]

        self.learning_curves_fig = config["perceptron_hyper"]["learning_curves_fig"].format(self.n_epochs, self.lr)
        
        # load the Iris data
        logger.info(">>> Loading the Iris dataset...")
        start_prep = time.time()
        self.X, self.y = load_iris_binary_data(csv_file = self.csv_file,
                features = self.features, pos_class = self.pos_class, neg_class = self.neg_class, logger = logger)

        assert(self.X is not None)
        assert(self.y is not None)

        end_prep = time.time()
        logger.info("\tData loaded in {} seconds".format(end_prep - start_prep))

        logger.info(">>> self.X.shape = {}, self.y.shape = {}".format(self.X.shape, self.y.shape))


    #########################################
    ## test_plot_learning_curves_scores()  ##
    #########################################

    def test_plot_learning_curves_scores(self):
 
        assert(logger is not None)
        
        # the perceptron
        self.perceptron = Perceptron(lr = self.lr, n_epochs = self.n_epochs, ascii = True)
       
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("########################################")
        logger.info("## test_plot_learning_curves_scores() ##")
        logger.info("########################################")
        logger.info("")

        start_lc = time.time()
        plot_learning_curves_cv_scores(self.perceptron,
                                       self.X, self.y,
                                       cv = 5,
                                       title = "Learning curves for Perceptron - epochs = {} - lr = {}".format(self.n_epochs, self.lr),
                                       logger = logger).savefig(self.learning_curves_fig, dpi=300)
 
        end_lc = time.time()
        logger.info("Learning curves saved in file  {} seconds".format(self.learning_curves_fig))
        logger.info("Learning curves computed in {} seconds".format(end_lc - start_lc))

        logger.info("End of test_plot_learning_curves_acc_score()")

    ###########################################
    ## test_plot_learning_curves_scores_2()  ##
    ###########################################

    def test_plot_learning_curves_scores_2(self):
 
        assert(logger is not None)
        
        # the perceptron
        self.perceptron = Perceptron(lr = self.lr, n_epochs = self.n_epochs, ascii = True)
       
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("##########################################")
        logger.info("## test_plot_learning_curves_scores_2() ##")
        logger.info("##########################################")
        logger.info("")
        logger.info("Draw only: learning curves + fit times vs data (e.g. scores_vs_fit_times = False)...")

        self.learning_curves_fig = self.learning_curves_fig[0:-4] + "_2.png" 
        
        start_lc = time.time()
        plot_learning_curves_cv_scores(self.perceptron,
                                       self.X, self.y,
                                       cv = 5,
                                        scores_vs_fit_times = False,
                                       title = "Learning curves for Perceptron - epochs = {} - lr = {}".format(self.n_epochs, self.lr),
                                       logger = logger).savefig(self.learning_curves_fig, dpi=300)
 
        end_lc = time.time()
        logger.info("Learning curves saved in file  {} seconds".format(self.learning_curves_fig))
        logger.info("Learning curves computed in {} seconds".format(end_lc - start_lc))

        logger.info("End of test_plot_learning_curves_acc_score_2()")

    ###########################################
    ## test_plot_learning_curves_scores_3()  ##
    ###########################################

    def test_plot_learning_curves_scores_3(self):
 
        assert(logger is not None)
        
        # the perceptron
        self.perceptron = Perceptron(lr = self.lr, n_epochs = self.n_epochs, ascii = True)
       
        assert(self.perceptron is not None)

        logger.info("")
        logger.info("##########################################")
        logger.info("## test_plot_learning_curves_scores_3() ##")
        logger.info("##########################################")
        logger.info("")
        logger.info("Draw only: learning curves + scores vs fit times (e.g. fit_times_vs_data = False)...")

        self.learning_curves_fig = self.learning_curves_fig[0:-4] + "_3.png" 
        
        start_lc = time.time()
        plot_learning_curves_cv_scores(self.perceptron,
                                       self.X, self.y,
                                       cv = 5,
                                        fit_times_vs_data = False,
                                       title = "Learning curves for Perceptron - epochs = {} - lr = {}".format(self.n_epochs, self.lr),
                                       logger = logger).savefig(self.learning_curves_fig, dpi=300)
 
        end_lc = time.time()
        logger.info("Learning curves saved in file  {} seconds".format(self.learning_curves_fig))
        logger.info("Learning curves computed in {} seconds".format(end_lc - start_lc))

        logger.info("End of test_plot_learning_curves_acc_score_3()")


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
    sys.argv[:] = argv

    with open(flags.conf, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    assert(config is not None)

    logger = init_logger(name="test_plotters", config = config)

    logger.info("")
    logger.info("######################")
    logger.info("## test_plotters.py ##")
    logger.info("######################")
    logger.info("")

    unittest.main()


