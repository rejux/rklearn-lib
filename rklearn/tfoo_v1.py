#!/usr/bin/env python
# -*- coding: utf-8 -*-

################
## tfoo_v1.py ##
################

# The following classes are provided by this single module framework:
# - BaseModel
# - BaseTrainer
# - BaseDataGenerator 
# - TensorBoardLogger

# The following tooling functions are also provided:



# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
from bunch import Bunch
import numpy as np ; np.random.seed(seed = 1)
import os
import time
# import sys 

# rktools
from rktools.monitors import ProgressBar

# TensorFlow v1.x
import tensorflow as tf

#################################################################################################
##                                          GPUManager                                         ##
#################################################################################################

class GPUManager:

    def __init__(self, logger = None):
        try:
            self.physical_gpus = tf.config.experimental.list_physical_devices('GPU')
            


        except Exception as e:
           raise RuntimeError(e) 

        self.logger = logger
        




#################################################################################################
##                                           BaseGraph                                         ##
#################################################################################################

class BaseGraph:

    ###################
    ## reset_graph() ##
    ###################

    @staticmethod
    def reset_graph(seed=42):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

#################################################################################################
##                                          BaseDataGenerator                                  ##
#################################################################################################

class BaseDataGenerator:

    ################
    ## __init__() ##
    ################
 
    def __init__(self, config, logger = None):
        self.config = config
        self.logger = logger
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.num_iter_per_epoch = None 
        self.batch_size = -1

    #######################
    ## get_data_shapes() ##
    #######################

    def data_shapes(self):
        ret = dict()
        ret["train"] = (self.X_train.shape, self.y_train.shape)
        ret["test"] = (self.X_test.shape, self.y_test.shape)

        if self.X_val is not None:
            ret["val"] = (self.X_valid.shape, self.y_valid.shape)
        
        return ret

    ##################
    ## next_batch() ##
    ##################
 
    def next_batch(self, shuffle = True):
        """
        Shuffle the training data and provide the next batch of training data. 
        This method should be a generator. 
        
        Ex. 

        TODO 

        Parameters:
            ...
            
        Returns:
            the next batch of training data
        """
     
        if self.X_train is None or self.y_train is None:
            raise NotImplementedError

        # the first time we have to calculate it
        # the number of iterations per epoch depends on the size of the batch
        if self.num_iter_per_epoch is None:             
            self.num_iter_per_epoch = len(self.X_train) // self.batch_size 
 
        if shuffle:
            rnd_idx = np.random.permutation(len(self.X_train))
        else:
            rnd_idx = [i for i in range(0, len(self.X_train) + 1)] 

        for batch_idx in np.array_split(rnd_idx, self.num_iter_per_epoch):
            X_batch, y_batch = self.X_train[batch_idx], self.y_train[batch_idx]
            yield X_batch, y_batch


    ## ============================ Override the following methods ============================= ##

    #################
    ## load_data() ##
    #################
 
    def load_data(self):
        raise NotImplementedError
    
    ####################
    ## prepare_data() ##
    ####################
 
    def prepare_data(self):
        raise NotImplementedError
 
#################################################################################################
##                                           BaseModel                                         ##
#################################################################################################

class BaseModel:

    ################
    ## __init__() ##
    ################

    def __init__(self, config, logger = None):
        """
        Parameters:
            - config: a dict
                ...
            - logger: an object
                ...
        Returns:
            Construct the object
        """

        self.config = config
        self.logger = logger
        self.X = None           # input data
        self.y = None           # input labels
        self.checkpoint_saver = None

        # get the following parameters from the config: 
        self.num_epochs = -1
        self.learning_rate = -1
        self.checkpoint_dir = None
        self.model_dir = None
        self.max_to_keep = -1

        # self.init_checkpoint_saver()

    #############################
    ## init_checkpoint_saver() ##
    #############################

    def init_checkpoint_saver(self):
        """
        Ensure the self.config object has a max_to_keep property
        """

        if self.max_to_keep < 0:
            self.max_to_keep = 5

        self.checkpoint_saver = tf.compat.v1.train.Saver(max_to_keep = self.max_to_keep)
        
    ########################
    ## save_checkpoints() ##
    ########################

    def save_checkpoints(self, sess):
        """
        Save function that saves the checkpoints in the path defined in the config file.

        Parameters:
            ...
        
        Returns:
            ...

        """

        if self.logger:
            self.logger.info("Saving model checkpoints...")

        if self.checkpoint_saver:
            # self.checkpoint_saver.save(sess, self.checkpoint_dir, self.global_step_tensor)
            self.checkpoint_saver.save(sess, self.checkpoint_dir)
        else:
            raise RuntimeError("self.checkpoint_saver is None!")

    ##############################
    ## load_latest_checkpoint() ##
    ##############################

    def load_latest_checkpoint(self, sess):
        """
        Load latest checkpoint from the experiment path defined in the config file

        Parameters:
            ...
        
        Returns:
            ...

        """

        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            if self.logger: 
                self.logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
    
    ## ============================ Override the following methods ============================= ##

    ###################
    ## build_model() ##
    ###################

    def build_model(self):
        """
        The architecture of the model is defined here!
        """
        raise NotImplementedError

#################################################################################################
##                                          TensorBoardLogger                                  ##
#################################################################################################

class TensorBoardLogger:
 
    ################
    ## __init__() ##
    ################
    
    def __init__(self, config):
        # self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = \
            tf.compat.v1.summary.FileWriter(os.path.join(self.config.summary_dir.format(os.getcwd()), "train"), 
                                            graph = tf.get_default_graph())
        self.test_summary_writer = \
            tf.compat.v1.summary.FileWriter(os.path.join(self.config.summary_dir.format(os.getcwd()), "test"))

    #################
    ## summarize() ##
    #################
 
    # it can summarize scalars and images.
    def summarize(self, sess, step, summarizer="train", scope="", summaries_dict=None):
        """
        Parameters:
            step: 
                the step of the summary
            summarizer: 
                use the train summary writer or the test one
            scope: 
                variable scope
            summaries_dict: 
                the dict of the summaries values (tag,value)
        Returns:
            Nothing
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.compat.v1.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()


#################################################################################################
##                                          BaseTrainer                                        ##
#################################################################################################

class BaseTrainer:
    
    def __init__(self, sess, model, data_gen, config, tb_logger = None, logger = None):
        """
        Construct the Trainer

        Parameters:
            ...
        
        Returns:
            ...
        """
        self.model = model
        self.tb_logger = tb_logger
        self.config = config
        self.sess = sess
        self.data_gen = data_gen
        self.logger = logger
        self.var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    
    ##################
    ## train_step() ##
    ##################
    
    def train_step(self):        
        """
        Implement the logic of one train step for a given epoch: 
            - process all batchs of data
            - evaluate the loss, and the metrics for all batchs

        Parameters:
            ...
        
        Returns:
            ...


        """
        # p_batch = ProgressBar(max_value=self.data_gen.num_iter_per_epoch, desc = "Batch: ", ascii = True)
        
        for batch_X, batch_y in self.data_gen.next_batch():
            _, loss, acc = self.sess.run([self.model.training_op, self.model.loss, self.model.accuracy], 
                                         feed_dict={self.model.X: batch_X,
                                                    self.model.y: batch_y,
                                                    self.model.train: True})
            # p_batch.update(1)
            
        # p_batch.close()
        
        return loss, acc # , step
            
    
    #############
    ## train() ##
    #############
    
    def train(self):        
        """
        Implement the train logic:
            - loop on the number of epochs, and for each epoch execute the step
            - add any summaries you want using the summary logger



        """
       
        if self.model.num_epochs == -1 or self.model.learning_rate == -1:
            raise NotImplementedError

        start = time.time()

        self.losses = []
        self.accs = []      # contains tuples of accuracies: (train, validation)

        p_epoch = ProgressBar(max_value = self.model.num_epochs, desc="Epoch: ", ascii = True)
        for epoch in range(1, self.model.num_epochs + 1):

            # execute the step here: iterate on batchs and get the loss and train acc 
            loss, acc_train = self.train_step()
                        
            # eval validation/testing accuracy on validation or testing set
            acc_valid = self.model.accuracy.eval(session=self.sess, feed_dict={self.model.X: self.data_gen.X_test,
                                                                               self.model.y: self.data_gen.y_test,
                                                                               self.model.train: False})
            
            # keep losses and accs for further analysis
            self.accs.append((acc_train,acc_valid))
            self.losses.append(loss)
                        
            # log on console & update progress
            msg = "Epoch = {}/{}, "
            msg += "train acc = {:.2f}, "
            msg += "val acc = {:.2f}"
            msg = msg.format(epoch, self.model.num_epochs, acc_train, acc_valid)
            
            # p_epoch.write(msg)
            p_epoch.p.set_description(msg)
            p_epoch.p.refresh()
            p_epoch.update(1)
                    
        p_epoch.close()
 
        end = time.time()
        if self.logger:
            self.logger.debug("train() performed in {} secs.".format((end - start)))


    ########################
    ## eval_performance() ##
    ########################

    def eval_performance(self, metric = "accuracy"):
        """
        Evaluate the model performance using the current session, graph and model
        
        Returns:
        --------
            double: the model performance (see the model.build_model() method)

        """
        if metric == "accuracy":
            return self.model.accuracy.eval(session=self.sess, feed_dict={self.model.X: self.data_gen.X_test,
                                                                          self.model.y: self.data_gen.y_test,
                                                                          self.model.train: False})
        else:
            raise NotImplementedError("The requested metric = {} is not implemented!")







