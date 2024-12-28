import numpy as np
from MLP import MLP

class AutoEncoder:
    def __init__(self, input_size, hidden_layers=1, neurons_per_layer=[16],
                 activation='relu', optimizer='sgd', learning_rate=0.01, batch_size=32, epochs=100):
        # Initialize the AutoEncoder
        self.input_size = input_size
        # self.latent_size = latent_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.mlp = MLP(input_size, input_size, regression=True, 
                       hidden_layers=hidden_layers, 
                       neurons_per_layer=neurons_per_layer,
                       activation=activation, 
                       optimizer=optimizer, 
                       learning_rate=learning_rate, 
                       batch_size=batch_size, 
                       epochs=epochs)

    def fit(self, X):
        self.mlp.fit(X, X)

    def get_latent(self, X):
        activations = self.mlp.getAllActivations(X)
        return activations