import numpy as np
from abc import abstractmethod

import tensorflow as tf


class DaggerAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.model = None

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        self.model.train(data_batch, label_batch)

    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict


# here is my own Agent
class MyDaggerAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        my_feature_columns = [tf.feature_column.numeric_column(key="obs")]
        self.model = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            hidden_units=[10, 10],
            n_classes=necessary_parameters)

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        data = []
        for i in data_batch:
            for j in i:
                for k in j:
                    data.append(k)
        data_dict = {"obs": data}
        self.model.train(input_fn=lambda: MyDaggerAgent.train_input_fn(
            self, data_dict, label_batch), steps=100)

    # select actions by your model
    def select_action(self, data_batch):
        data = []
        for i in data_batch:
            for j in i:
                for k in j:
                    data.append(k)
        data_dict = {"obs": data}
        predictions = self.model.predict(input_fn=lambda: MyDaggerAgent.eval_input_fn(self, data_dict))
        id = 0
        predictions = list(predictions)
        for pred in predictions:
            id = pred['class_ids'][0]
        return id

    def train_input_fn(self, features, labels):
        return tf.data.Dataset.from_tensor_slices((dict(features), labels)).batch(1)

    def eval_input_fn(self, features):
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(1)
