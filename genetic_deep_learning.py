"""
Title: Genetic Deep Learning 
Description: Sets up a Deep Neural Network model as target predictive model for genetic optimization 
"""

#
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
#
from genetic_optimization import ServerModelConfOptimizer

#
class ServerDNNOptimizer(ServerModelConfOptimizer):

    DEFAULT_PARAMS = {
        'epochs': ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],
        'batch_size': ['16', '32', '64', '128', '256', '312', '384', '448', '512', '576'],
        'hidden_layer_size': ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500'],
        'embedding_layer_size': ['50', '100', '150', '200', '250', '300', '350', '400', '450', '500'],
        'weight_initialization': ['he_normal', 'glorot_normal', 'normal'],
        'clipvalue': ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
        'learning_rate': ['0.001', '0.002', '0.003', '0.004', '0.005', '0.006', '0.007', '0.008', '0.009', '0.01'],
        'neuron_activation': ['relu', 'tanh', 'sigmoid']
    }


    def fit_predict(self, individual):
        tf.get_logger().setLevel('ERROR')

        # Recast numeric parameters from string to appropriate numeric type
        epochs = int(individual['epochs'])
        batch_size = int(individual['batch_size'])
        hidden_layer_size = int(individual['hidden_layer_size'])
        embedding_layer_size = int(individual['embedding_layer_size'])
        clipvalue = float(individual['clipvalue'])
        learning_rate = float(individual['learning_rate'])

        # Extract categorical parameters
        weight_init = individual['weight_initialization']
        neuron_activation = individual['neuron_activation']

        # Data normalization
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled_train = input_scaler.fit_transform(self.train_X)
        x_scaled_test = input_scaler.transform(self.test_X)

        y_train = self.train_Y.reshape(self.train_Y.shape[0], 1)
        response_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled_train = response_scaler.fit_transform(y_train)

        # Define Model Architecture
        model = Sequential()
        model.add(Dense(embedding_layer_size, activation=neuron_activation, kernel_initializer=weight_init, input_shape=[x_scaled_train.shape[1]]))
        model.add(Dense(hidden_layer_size, activation=neuron_activation))
        model.add(Dense(1, activation="linear"))

        # Compile model with specified learning rate and clipvalue
        optimizer = optimizers.Adam(learning_rate=learning_rate, clipvalue=clipvalue)
        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_absolute_error"])

        # Train Model
        model.fit(x_scaled_train, y_scaled_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

        # Generate Predictions
        predictions_scaled = model.predict(x_scaled_test)

        # Inverse scale predictions to get forecast in original scale
        predictions = response_scaler.inverse_transform(predictions_scaled)
        forecast = predictions.flatten()

        return forecast
