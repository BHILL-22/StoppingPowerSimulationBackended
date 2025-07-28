#!/usr/bin/env python
# coding=utf-8
from stopping_power_ml.rc import *
import keras
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 

class Cast(keras.layers.Layer):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        if isinstance(dtype, str):
            try:
                self.target_dtype = jnp.dtype(dtype)
            except TypeError: # Fallback if jnp.dtype() doesn't like the string directly
                 if dtype.lower() == 'float32': self.target_dtype = jnp.float32
                 elif dtype.lower() == 'float64': self.target_dtype = jnp.float64
                 elif dtype.lower() == 'int32': self.target_dtype = jnp.int32
                 else: raise ValueError(f"Unknown dtype string for JAX: {dtype}")
        else:
            self.target_dtype = dtype 
        logging.info(f"target dtype: {self.target_dtype}")


    def call(self, inputs):
        return jnp.astype(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": str(self.target_dtype)})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_fn(input_size=10, dense_layers=(10,10), activation='linear', use_linear_block=True,
            optimizer_options=dict(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mean_absolute_error'])):
    """Creates a Keras NN model
    
    Args:
        input_size (int) - Number of features in input vector
        dense_layers ([int]) - Number of units in the dense layers
        activation (str) - Activation function in the dense layers
        use_linear_block (bool) - Whether to use the linear regression block
        optimizer_options (dict) - Any options for the optimization routine
    Returns:
        (keras.models.Model) a Keras model
    """
    logging.info(f"kera backend {keras.backend.backend()}")
    logging.info(f"jax device {jax.devices()}")

    # Input layer
    inputs = keras.layers.Input(shape=(input_size,), name='input', dtype = 'float64')
    
    # Make the dense layer
    dense_layer = inputs
    for layer_size in dense_layers:
        dense_layer = keras.layers.Dense(layer_size, activation=activation)(dense_layer)
    
    if use_linear_block:
        # Add the LR layer
        combined_layer = keras.layers.concatenate([dense_layer, inputs])
    else:
        combined_layer = dense_layer
    
    # Output layer
    outputs = keras.layers.Dense(1, activation='linear', name='output')(combined_layer)
    
    # Make/compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(**optimizer_options)
    return model

