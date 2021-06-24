import numpy as np
import tensorflow as tf
from tensorflow import keras

from tqdm import tqdm
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import History
from tensorflow.keras import Sequential
from typing import Tuple

def create_model(X: np.ndarray, Y: np.array) -> Tuple[Sequential, History]:
    """Creates a SimpleRNN-based model for the adding problem.

    Args:
        X (np.ndarray): Training data consisting of (float, mask-value)
            pairs for each time step
        Y (np.array): Response/target variable

    Returns:
        Tuple[Sequential, History]: Trained model and History object
    """

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(100,
                            return_sequences=True,
                            input_shape=[None, 2],
                            activation='relu',
                            recurrent_initializer='orthogonal'
                            #recurrent_initializer='identity'
                            ),
        keras.layers.SimpleRNN(1, activation='linear')
    ])

    #optimizer = keras.optimizers.Adam(learning_rate=0.0002, epsilon=0.5)#, clipvalue=10)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, epsilon=0.01)#, clipvalue=10)
    model.compile(loss="mse", optimizer=optimizer)
    history = model.fit(X,
                        Y,
                        epochs=32,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[TqdmCallback(verbose=0, tqdm_class=tqdm, desc='Train model')])

    return model, history
