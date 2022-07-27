#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Perform inferences by predicting using the test data from either saved model or model server.
Savedmodel is trained keras model(.pb) saved to disk whereas model server is Tensorflow Serving which 
deploys trained models on the server. For comparison, this class provides functionality to infer from
the model that is either saved trained model or the same model deployed on TF serving(in my case: localhost)
using the same test data. As a future step, localhost can be replaced with actual cloud server by hosting
deployed model into the cloud.
"""

from typing import Any, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import requests
import json
import utilities.config as config
from utilities.utils import Dataset, Preprocess, create_dir

class Inference:
   def __init__(self, x_test: Dataset, model_dir: str, prediction_source: str,  
               api_endpoint_url: str = None) -> Any:

      self.x_test = x_test
      self.model_dir = model_dir
      self.prediction_source = prediction_source
      self.api_endpoint_url = api_endpoint_url
      """
      Parameters
      ----------
      x_test: Dataset
         Preprocessed data for evaluation.
      model_dir: str
         Directory to load saved model.
      prediction_source: str
         Name of the prediction service (Eg: savedmodel or modelserver).
      api_endpoint_url: str = None
         URL of endpoint where the model is deployed in the modelserver.
         It is obtained once the model is served to tf serving by docker
         from the terminal. Defaults to None.
      """

   def predict_from_savedModel(self) -> np.array:
      """
      Load and predict from saved keras model.

      Returns
      -------
      model predictions: np.array
      """
      print("Predicting from saved model .........")
      savedModel = tf.keras.models.load_model(self.model_dir)
      savedModel.summary()
      return savedModel.predict(self.x_test[0:10])

   def predict_from_modelServer(self) -> Dict[list]:
      """
      Predicting from the model server (tf serving) deployed in localhost.

      Returns
      -------
      predictions: Dict[list]
         Model predictions obtained from the server as JSON response.
      """
      print(f"Predicting from access endpoint URL: {self.api_endpoint_url}")
      input_data = json.dumps({"signature_name": "serving_default", "instances": self.x_test.tolist()})
      json_response = requests.post(self.api_endpoint_url, data= input_data)
      predictions = json.loads(json_response.text)['predictions']
      return predictions

   #Tip: Replacing if/else with python dicts.
   PREDICTION_SOURCES: dict = {
      "predict_from_savedmodel": predict_from_savedModel,
      "predict_from_modelserver": predict_from_modelServer,
   }

   def predict(self) -> Any:
      return self.PREDICTION_SOURCES[self.prediction_source]()

if __name__ == "__main__":
   model_dir, _ = create_dir('saved_models', config.method_name)
   (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
   dataset_ = Dataset(x_tr, y_tr, x_te, y_te)
   preprocessed_dataset_: Dataset = Preprocess(dataset_).preprocess_normal_dataset()
   predictions: Any = Inference(preprocessed_dataset_.x_test,
                                model_dir, 
                                config.PREDICTION_SOURCE, 
                                config.API_ENDPOINT_URL).predict()

   for true_value, predicted_value in enumerate(predictions):
      print(f"True Value: {preprocessed_dataset_.y_test[true_value]} and Predicted Value: {np.argmax(predicted_value)}")