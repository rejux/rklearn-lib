#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############
## Imports ##
#############

import os
import sys ; sys.path.append("/home/developer/workspace/rklearn-lib")
import time
import pickle
import numpy as np

from rklearn.tfoo_v1 import BaseDataGenerator

from rktools.monitors import ProgressBar


############################
## CIFAR10DataGenerator() ##
############################

class CIFAR10DataGenerator(BaseDataGenerator):
    
    ################
    ## __init__() ##
    ################

    def __init__(self, config, logger = None):
        try: 
            super().__init__(config, logger)
            self.logger = logger
            self.config = config 
            self.data_top_dir = self.config.data["data_home"]
            self.batch_size = self.config.data["batch_size"]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            raise RuntimeError("error msg = {}, error type = {}, error file = {}, error line = {}".format(e, exc_type, fname, exc_tb.tb_lineno))
        

    #################
    ## load_data() ##
    #################

    def load_data(self):
        """
        Load both training and testing data.
        """

        if not os.path.exists(self.data_top_dir):
            raise FileNotFoundError("Directory {} is not valid!".format(self.data_top_dir))

        try:

            start = time.time()

            # Read CIFAR training data

            nb_files = self.config.data["train_data_nb_files"]
            progress_bar = ProgressBar(max_value = nb_files, desc="File: ", ascii = True) 
            for file_index in range(nb_files):

                file_path = os.path.join(self.data_top_dir, self.config.data["train_data_batch_prefix"] + str(file_index+1))
                assert(os.path.exists(file_path))
                train_file = open(file_path, "rb")
                train_dict = pickle.load(train_file, encoding="latin1")
                train_file.close()

                # 1st file
                if self.X_train is None:
                    self.X_train = np.array(train_dict['data'], float) 
                    self.y_train = train_dict['labels']
                else:
                    self.X_train = np.concatenate((self.X_train, train_dict["data"]), 0)
                    self.y_train = np.concatenate((self.y_train, train_dict["labels"]), 0)
                
                progress_bar.update(1)

            progress_bar.close()

            # Read CIFAR test data
            
            file_path = os.path.join(self.data_top_dir, self.config.data["test_data_batch_prefix"])
            assert(os.path.exists(file_path))
            test_file = open(file_path, "rb")
            test_dict = pickle.load(test_file, encoding="latin1")
            test_file.close()
            self.X_test = test_dict["data"]
            self.y_test = np.array(test_dict["labels"])
            
            # for dev
            if self.config.data["dev_sample"] >0:

                train_sample_size = int(len(self.X_train) * self.config.data["dev_sample"]) 
                self.X_train = self.X_train[:train_sample_size] 
                self.y_train = self.y_train[:train_sample_size]
                
                test_sample_size = int(len(self.X_train) * self.config.data["dev_sample"])
                self.X_test = self.X_test[:test_sample_size]
                self.y_test = self.y_test[:test_sample_size]

            end = time.time()
            if self.logger:
                self.logger.debug("CIFAR 10 data loaded in {} secs.".format((end - start)))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            raise RuntimeError("error msg = {}, error type = {}, error file = {}, error line = {}".format(e, exc_type, fname, exc_tb.tb_lineno))

    ####################
    ## prepare_data() ##
    ####################

    def prepare_data(self): 

        start = time.time()

        # Preprocess training data and labels
        self.X_train = self.X_train.astype(np.float32) / 255.0 # normalize 
        self.X_train = self.X_train.reshape([-1, self.config.data["num_channels"], 
                                             self.config.data["image_size"],
                                             self.config.data["image_size"]])
        self.X_train = self.X_train.transpose([0, 2, 3, 1])
        self.y_train = np.eye(self.config.data["num_categories"])[self.y_train]

        # Preprocess test data and labels
        self.X_test = self.X_test.astype(np.float32) / 255.0 # normalize 
        self.X_test = self.X_test.reshape([-1, self.config.data["num_channels"], 
                                           self.config.data["image_size"], 
                                           self.config.data["image_size"]])
        self.X_test = self.X_test.transpose([0, 2, 3, 1])
        self.y_test = np.eye(self.config.data["num_categories"])[self.y_test]

        end = time.time()
        if self.logger:
            self.logger.debug("Data prepared in {} secs.".format((end - start)))

