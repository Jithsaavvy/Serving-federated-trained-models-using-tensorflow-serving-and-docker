#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Training pipeline for both FL and NON-FL models.

Note:
    The functionality to save tff models is still not a part of PIP and it is 
    only given in tensorflow research page. So, due to this reason, it is not 
    possible to include the code snippet of tff model saving part. For more 
    information, refer: https://github.com/google-research/federated/tree/master/gans 
"""

import os
import typing
import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import tensorflow_federated as tff
import config
from utils import Model, Dataset, FederatedDataset, Metrics, init_function, utility

class Training:
    def __init__(self, model: Model, dataset: Dataset, federatedDataset: FederatedDataset, 
                model_dir: str, strategy: str) -> None:
        """
        Parameters
        ----------
        model: Model
            Input model to train.
        dataset: Dataset
            Instance of Dataset class holding the processed data(train & test).
            This is required for both FL and NON-FL training.
        federatedDataset: FederatedDataset
            Instance of FederatedDataset holding processed data from dataset for FL.
        model_dir: str
            Directory to save model after training.
        strategy: str
            Defines the type of training. Distinguishes between FL & NON-FL training.
            (Eg: normal for normal training, fl_random for fl training with random
            initialized model & fl_pretrained for fl training with pretrained model).
            Assign in config.STRATEGY in config.py

        Returns
        -------
            None
        """
        self.model = model
        self.dataset_ = dataset
        self.federatedDataset_ = federatedDataset
        self.model_dir = model_dir
        self.strategy = strategy

    def normal_convnet_training(self) -> Metrics:
        """
        Method for normal convnet training.

        Returns
        -------
        instanceof(Metrics):
            Instance will hold metrics information for plotting, analyzing &
            further usage.
        """
        self.model.compile(loss=losses.sparse_categorical_crossentropy,
                        optimizer=optimizers.SGD(learning_rate=config.LEARNING_RATE),
                        metrics=['accuracy'])
        history = self.model.fit(self.dataset_.x_train, self.dataset_.y_train, 
                batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=1,
                validation_data=(self.dataset_.x_test, self.dataset_.y_test))
        train_accuracy = list(np.array(history.history['accuracy']))
        train_loss = list(np.array(history.history['loss']))
        val_accuracy = list(np.array(history.history['val_accuracy']))
        val_loss = list(np.array(history.history['val_loss']))
        save_path = os.path.join(config.model_dir, config.model_name)
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

        return Metrics(train_accuracy, train_loss, val_accuracy, val_loss)

    
    def federated_convnet_training_random(self) -> Metrics:
        """
        Federated training of random initialization model.

        Returns
        -------
        instanceof(Metrics):
            Instance will hold metrics information for plotting, analyzing &
            further usage.

        Refer above documentation regarding tff model saving!!!
        """
        init_fn = init_function(self.model, self.dataset_.processed_data)
        iterative_process = tff.learning.build_federated_averaging_process(
                    init_fn,
                    client_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    server_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    use_experimental_simulation_loop=True)
        
        state = iterative_process.initialize()
        print("FL Training with random initialization started...")
        metrics_instance: Metrics = utility(self.model, state, iterative_process, self.federatedDataset_.federated_train_data,
                                    self.dataset_.x_test, self.dataset_.y_test)
        return metrics_instance

    def federated_convnet_training_pretrained(self) -> Metrics:
        """
        Federated training of pretrained models.

        Returns
        -------
        instanceof(Metrics):
            Instance will hold metrics information for plotting, analyzing &
            further usage.
        """
        pretrained_model = tf.keras.models.load_model(config.model_dir / config.model_name)
        init_fn = init_function(pretrained_model, self.dataset_.processed_data)
        iterative_process = tff.learning.build_federated_averaging_process(
                    init_fn,
                    client_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    server_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    use_experimental_simulation_loop=True)
        
        state = iterative_process.initialize()
        #Pushing the learned weights of pretrained model to the init state of FL model
        state = tff.learning.state_with_new_model_weights(
                    state,
                    trainable_weights=[v.numpy() for v in pretrained_model.trainable_weights],
                    non_trainable_weights=[
                    v.numpy() for v in pretrained_model.non_trainable_weights])
        print("FL Training with pretrained models started...")
        metrics_instance: Metrics = utility(pretrained_model, state, iterative_process, self.federatedDataset_.federated_train_data,
                                    self.dataset_.x_test, self.dataset_.y_test)
        return metrics_instance

    #Tip: Replacing if/else with python dicts.
    TRAINING_STRATEGIES: dict = {
            "normal": normal_convnet_training,
            "fl_random": federated_convnet_training_random,
            "fl_pretrained": federated_convnet_training_pretrained,
            }

    def train(self) -> Metrics:
        return self.TRAINING_STRATEGIES[self.strategy]()