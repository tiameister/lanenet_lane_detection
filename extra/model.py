import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
from keras import backend 
from keras import optimizers
from keras import models


#tf.python.control_flow.control_flow_ops = tf
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection
from data import generate_samples, preprocess
from weights_logger_callback import WeightsLogger

local_project_path = '/home/taha/catkin_ws/src/behavioral_cloning/data'
batch_size = 12
local_data_path = os.path.join(local_project_path)


if __name__ == '__main__':
    # Read the data
    df = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
    # Split data into training and validation sets
    df_train, df_valid = model_selection.train_test_split(df, test_size=.2)

    # Model architecture
    model = models.Sequential()
    model.add(keras.layers.Conv2D(16, (3, 3), input_shape=(32, 128, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(20, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

    history = model.fit_generator(
        generate_samples(df_train, local_data_path),
        steps_per_epoch=len(df_train) // batch_size,
        epochs=30,
        validation_data=generate_samples(df_valid, local_data_path, augment=False),
        validation_steps=len(df_valid) // batch_size,
        callbacks=[WeightsLogger(root_path=local_project_path)]
    )

    backend.clear_session()