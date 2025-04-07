import numpy as np
from multi_label_metrices import *


class MultiLabelMLP:
    def __init__(self, input_size, output_size, hidden_layers, class_labels, learning_rate=0.01, 
                 activation_function='relu', optimizer='sgd', batch_size=32, epochs=100):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.class_labels = class_labels
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.weights = []
        self.biases = []
        self.activations = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
            self.biases.append(np.zeros((1, layers[i + 1])))
    
    def _activation(self, Z):
        if self.activation_function == 'sigmoid':
            return self._sigmoid(Z)
        elif self.activation_function == 'tanh':
            return self._tanh(Z)
        elif self.activation_function == 'relu':
            return self._relu(Z)
        elif self.activation_function == 'linear':
            return self._linear(Z)
        else:
            raise ValueError(f"Invaild activation function {self.activation_function}")
        
    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _tanh(self, Z):
        return np.tanh(Z)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _linear(self, Z):
        return Z
    
    def _activation_derivative(self, A):
        if self.activation_function == 'sigmoid':
            return self._sigmoid_derivative(A)
        elif self.activation_function == 'tanh':
            return self._tanh_derivative(A)
        elif self.activation_function == 'relu':
            return self._relu_derivative(A)
        elif self.activation_function == 'linear':
            return self._linear_derivative(A)
        else:
            raise ValueError(f"Invaild activation function {self.activation_function}")
        
    def _sigmoid_derivative(self, A):
        return A * (1 - A)
    
    def _tanh_derivative(self, A):
        return 1 - A ** 2

    def _relu_derivative(self, A):
        return (A > 0).astype(float)
    
    def _linear_derivative(self, A):
        return np.ones_like(A)
    
    def forward_propagation(self, X):
        self.activations = [X]
        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._activation(Z)
            self.activations.append(A)
        
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self._sigmoid(Z)
        self.activations.append(A)
        
        return self.activations[-1]
    
    def backward_propagation(self, X, Y):
        grads_w = []
        grads_b = []
        m = X.shape[0]
    
        dZ = self.activations[-1] - Y 
        grads_w = []
        grads_b = []

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.activations[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            grads_w.insert(0, dW)
            grads_b.insert(0, dB)
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self._activation_derivative(A=self.activations[i])
        return grads_w, grads_b

    def _update_weights(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def _optimize(self, X_train, Y_train):
        if self.optimizer == 'sgd':
            self.stochastic_gradient_descent(X_train, Y_train)
        elif self.optimizer == 'batch-gd':
            self.batch_gradient_descent(X_train, Y_train)
        elif self.optimizer == 'mini-batch-gd':
            self.mini_batch_gradient_descent(X_train, Y_train)
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")

    def stochastic_gradient_descent(self, X_train, Y_train):
        for i in range(len(X_train)):
            X_sam = X_train[i:i+1]
            Y_sam = Y_train[i:i+1]
            self.forward_propagation(X_sam)
            grads_w, grads_b = self.backward_propagation(X_sam, Y_sam)
            self._update_weights(grads_w, grads_b)

    def batch_gradient_descent(self, X_train, Y_train):
        self.forward_propagation(X_train)
        grads_w, grads_b = self.backward_propagation(X_train, Y_train)
        self._update_weights(grads_w, grads_b)

    def mini_batch_gradient_descent(self, X_train, Y_train):
        for i in range(0, len(X_train), self.batch_size):
            X_batch = X_train[i:i+self.batch_size]
            Y_batch = Y_train[i:i+self.batch_size]
            self.forward_propagation(X_batch)
            grads_w, grads_b = self.backward_propagation(X_batch, Y_batch)
            self._update_weights(grads_w, grads_b)
    
    def binary_cross_entropy_loss(self, Y_pred, Y):
        m = Y.shape[0]
        loss = -1/m * np.sum(Y * np.log(Y_pred + 1e-8) + (1 - Y) * np.log(1 - Y_pred + 1e-8))
        return loss
    
    def fit(self, X_train, Y_train, validation_data=None):
        self.train(X_train, Y_train, validation_data)

    def train(self, X_train, Y_train, validation_data, patience=5):

        wandb.init(project="MLPMultiLabel_hyperparameter_tuning", config={
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_layers": self.hidden_layers,
            "activation_function": self.activation_function,
            "optimizer": self.optimizer,
        })
        
        early_stopping_counter = 0
        best_loss = float('inf')
        for epoch in range(self.epochs):
            self._optimize(X_train, Y_train)   

            if validation_data:
                X_val, Y_val = validation_data
                self.forward_propagation(X_val)
                val_loss = self.binary_cross_entropy_loss(self.activations[-1], Y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            self.forward_propagation(X_train)
            train_loss = self.binary_cross_entropy_loss(self.activations[-1], Y_train)
            y_pred = self.predict(X_train)
            train_evaluation = MultiLabelMetrics(y_true=Y_train, y_pred=y_pred)

            X_val, Y_val = validation_data
            self.forward_propagation(X_val)
            val_loss = self.binary_cross_entropy_loss(self.activations[-1], Y_val)
            y_pred = self.predict(X_val)

            val_evaluation = MultiLabelMetrics(y_true=Y_val, y_pred=y_pred)

            print(f"Epoch {epoch+1}/{self.epochs} complete, Train Loss : {train_loss}, Val Loss : {val_loss}")

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_evaluation.accuracy_score(),
                "val_accuracy": val_evaluation.accuracy_score(),
                "precision": val_evaluation.precision_score("macro"),
                "recall": val_evaluation.recall_score("macro"),
                "f1_score": val_evaluation.f1_score("macro"),
                "hamming_loss": val_evaluation.hamming_loss()
            })

        wandb.finish()

    def predict(self, X):
        self.forward_propagation(X)
        return (self.activations[-1] > 0.5).astype(int)  
    
    def gradient_check(self, X, Y, epsilon=1e-6):
        self.forward_propagation(X)
        grads_w, grads_b = self.backward_propagation(X, Y)

        for weight, bias, gradw, gradb in zip(self.weights, self.biases, grads_w, grads_b):
            num_gradw = np.zeros_like(weight)
            num_gradb = np.zeros_like(bias)

            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    original = weight[i, j]
                    weight[i, j] = original + epsilon
                    self.forward_propagation(X)
                    loss_plus_epsilon = self.binary_cross_entropy_loss(self.activations[-1], Y)

                    weight[i, j] = original - epsilon
                    self.forward_propagation(X)
                    loss_minus_epsilon = self.binary_cross_entropy_loss(self.activations[-1], Y)
                    weight[i, j] = original

                    num_gradw[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

            for i in range(bias.shape[1]):
                original = bias[0, i]
                bias[0, i] = original + epsilon
                self.forward_propagation(X)
                loss_plus_epsilon = self.binary_cross_entropy_loss(self.activations[-1], Y)

                bias[0, i] = original - epsilon
                self.forward_propagation(X)
                loss_minus_epsilon = self.binary_cross_entropy_loss(self.activations[-1], Y)
                bias[0, i] = original

                num_gradb[0, i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

            
            relative_difference = np.linalg.norm(gradw - num_gradw) / (np.linalg.norm(gradw + num_gradw))
            if relative_difference > 10-6:
                print(f"Gradient check failed: relative difference is {relative_difference}")
                return False
            
            relative_difference = np.linalg.norm(gradb - num_gradb) / (np.linalg.norm(gradb + num_gradb))
            if relative_difference > 10-6:
                print(f"Gradient check failed: relative difference is {relative_difference}")
                return False
       
        print("Gradient check passed!")
        return True