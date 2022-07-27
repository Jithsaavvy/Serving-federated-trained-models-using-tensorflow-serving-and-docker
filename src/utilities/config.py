"""
@author: Jithin Sasikumar

Configuration file that holds values for all variables used across the project. They include
directory names, training parameters, global variables used across files. These values are
subjective and could be changed based on the requirements and file structure. 
"""

#Training (FL & Non-FL) parameters
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 100
LEARNING_RATE = 1e-2
#For splitting dataset into no of clients
SPLIT = 100
img_rows, img_cols = 28, 28
INPUT_SHAPE = (img_rows, img_cols, 1)
EPOCHS_FL = 20
BATCH_SIZE_FL = 10
PREFETCH_BUFFER_FL = 10
NUM_ROUNDS = 100

"""
Name of models and other attributes. Some examples are,
    method_name: normal_training, fl_training_random, fl_training_pretrained
    model_name: normal, fl_random_trained, fl_pretrained
"""
method_name = "fl_training_random"
model_name = "fl_random_trained"
STRATEGY = "fl_random"
PLOT_TITLE = "Accuracy Plot - FL training with random initialization"
API_ENDPOINT_URL = "http://localhost:8602/v1/models/federated_model/versions/1:predict"
PREDICTION_SOURCE = "modelserver"