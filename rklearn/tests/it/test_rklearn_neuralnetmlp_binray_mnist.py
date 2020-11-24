#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################
## test_rklearn_neuralnetmlp_binray_mnist.py ##
###############################################






















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

    logger = init_logger(name="test_mnist_adalineXXX", config = config)

    logger.info("")
    logger.info("##########################################")
    logger.info("## test_rklearn_adaline_binary_mnist.py ##")
    logger.info("##########################################")
    logger.info("")

    main(config)

