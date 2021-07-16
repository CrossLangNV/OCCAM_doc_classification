"""
https://www.tensorflow.org/tutorials/images/transfer_learning
"""
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

IMAGE_WIDTH = 224

ROOT = os.path.join(os.path.dirname(__file__), '..')
FILENAME_MODEL = os.path.join(ROOT, r'models/mobilenet_imagenet.h5')


class DocModel(tf.keras.Model):
    """
    Document classifier, using a pretrained mobileNet.
    """

    def __init__(self):
        # Input shape (224, 224, 3)

        def get_base_model():

            if os.path.exists(FILENAME_MODEL):
                weights = FILENAME_MODEL
            else:
                weights = 'imagenet'

            base_model = MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, 3),
                                     include_top=False,
                                     weights=weights)

            if not os.path.exists(FILENAME_MODEL):
                # Download and save locally
                base_model.save_weights(FILENAME_MODEL)

            return base_model

        get_base_model()

        base_model = MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, 3),
                                 include_top=False,
                                 weights='imagenet')
        base_model.trainable = False  # Fix the weights for transfer learning

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1)  # Output considered a logit

        inputs = tf.keras.Input(shape=base_model.input_shape[1:])
        # preprocess [0-255] images
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        x = preprocess_input(inputs)
        f = base_model(x, training=False)

        input_f = tf.keras.Input(shape=f.shape[1:], name='classifier_input')
        y = global_average_layer(input_f)
        y = tf.keras.layers.Dropout(0.2)(y)
        outputs = prediction_layer(y)

        model_features = tf.keras.Model(inputs, f)
        model_classifier = tf.keras.Model(input_f, outputs)

        super(DocModel, self).__init__(model_features.inputs, model_classifier(model_features.outputs))

        # model = tf.keras.Model(model_features.inputs, model_classifier(model_features.outputs))

        def _compile(model):
            base_learning_rate = 0.0001
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['accuracy'])

        _compile(self)
        _compile(model_classifier)

        print(self.summary())

        self.base_model = base_model

        self.model_features = model_features
        self.model_classifier = model_classifier

    def fit_fast(self, x, y, validation_data: Tuple[np.ndarray, np.ndarray] = None, *args, **kwargs):
        """ Instead of inferring each input over and over again, the intermediate features are computed once and
        thus the last layer can be trained extremely fast.

        Args:
            x: list or array of shape (n, w, w, 3) containing the input images
            y: list or array of shape (n, 1) containing the ground truth class index.
            validation_data: Optional validation data. Behaves as (x, y).
            *args: Optional args
            **kwargs: Optional kwargs

        Returns:
            History object.
        """
        f = self.model_features(x)

        if validation_data is not None:
            x_val, y_val = validation_data
            f_val = self.model_features(x_val)
            validation_data_f = (f_val, y_val)
        else:
            validation_data_f = None

        return self.model_classifier.fit(f, y, validation_data=validation_data_f,
                                         *args, **kwargs)

    def feature(self, x):
        """ Returns a (n, 7, 7, 1280) array with n the batch size. n = 1 if a single image is given.

        Args:
            x:

        Returns:

        """
        if len(x.shape) == 3:
            x = x[np.newaxis]

        return self.base_model(x)
