#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Baseline normal training of convnet model using MNIST dataset without federated learning.
The script loads, preprocess data, creates model and train models followed by saving
the results & model respectively. 

Note: Change the variables accordingly in config.py based on the requirements.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from utilities.utils import Dataset, Model, Preprocess, Metrics, create_dir, save, plot
import utilities.config as config
from utilities.train import Training

np.random.seed(0)

if __name__ == "__main__":
    #Create directory for saving trained model and results
    model_dir, result_dir = create_dir('pretrained_models', config.method_name)

    #Load and prepare mnist dataset for training
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    dataset_ = Dataset(x_tr, y_tr, x_te, y_te)
    preprocess_ = Preprocess(dataset_)
    preprocessed_dataset_: Dataset = preprocess_.preprocess_normal_dataset()

    #Model creation & training
    model = Model(config.INPUT_SHAPE, config.NUM_CLASSES).build_convnet_model()
    metrics_: Metrics = Training(model, preprocessed_dataset_, model_dir, config.STRATEGY).normal_convnet_training()

    #Visualization & saving results to disk. Change the below config variables accordingly
    plot(list(range(1, 101)), metrics_.train_accuracy, config.PLOT_TITLE, "Train accuracy points", "Train accuracy")
    plot(list(range(1, 101)), metrics_.validation_accuracy, config.PLOT_TITLE, "Validation accuracy points", "Validation accuracy")
    save(metrics_, result_dir)