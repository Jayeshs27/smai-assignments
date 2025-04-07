import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../MLP")))

from MLP import MLP

class AutoEncoder:
    def __init__(self, 
                 input_dim, 
                 reduced_dim, 
                 encoder_hidden_layers, 
                 decoder_hidden_layers, 
                 activation_function='relu',
                 optimizer='sgd',
                 learning_rate=0.001, 
                 epochs=50, 
                 batch_size=32):

        hidden_layers = encoder_hidden_layers + [reduced_dim] + decoder_hidden_layers
        self.model = MLP(input_size=input_dim, 
                                   class_type='regression',
                                   output_size=input_dim, 
                                   hidden_layers=hidden_layers,
                                   activation_function=activation_function, 
                                   learning_rate=learning_rate, 
                                   optimizer=optimizer,
                                   epochs=epochs, 
                                   batch_size=batch_size)
        
        self.latent_dim = reduced_dim
        self.num_encoder_hidden_layers = len(encoder_hidden_layers)
        self.num_decoder_hidden_layers = len(decoder_hidden_layers)
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []

    def fit(self, X, X_val):
        self.model.fit(X, X, validation_data=(X_val, X_val))
        self.train_losses = self.model.train_losses
        self.val_losses = self.model.val_losses
    
    def get_latent(self, X):
        A = X
        for i in range(self.num_decoder_hidden_layers + 2):
            Z = np.dot(A, self.model.weights[i]) + self.model.biases[i]
            A = self.model._activation(Z)
            if A.shape[1] == self.latent_dim:
                return A
            
        print("No reduced layer found")
        return None

    def reconstruct(self, X):
        return self.model.predict(X)

    def mean_squared_error(self, Y_pred, Y_true):
        Y_true = Y_true.reshape(-1, Y_pred.shape[1])
        error = np.mean((Y_pred - Y_true) ** 2)
        return error