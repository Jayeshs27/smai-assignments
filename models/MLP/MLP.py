import numpy as np
import wandb 
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../performance-measures")))

from evaluation import model_evaluation

class MLP:
    def __init__(self, 
                 input_size,
                 output_size, 
                 hidden_layers, 
                 class_type='classification',
                 class_labels=None, 
                 learning_rate=0.01, 
                 activation_function='relu',
                 optimizer='sgd', 
                 batch_size=32, 
                 epochs=100, 
                 loss_function='mse',
                 last_layer_activation='linear',
                 ):
        self.class_type = class_type
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.class_labels = class_labels
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_function = loss_function
        self.last_layer_activation = last_layer_activation

        self.train_losses = []
        self.val_losses = []
        self.weights = []
        self.biases = []
        self.activations = []

    def _init_weights(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
            self.biases.append(np.zeros((1, layers[i+1])))

    def fit(self, X_train, Y_train, validation_data):
        self.train(X_train, Y_train, validation_data)

    def predict(self, X):
        if self.class_type == 'regression':
            return self.predict_proba(X)
        else:
            self.forward_propagation(X)
            predicted_indices = np.argmax(self.activations[-1], axis=1)
            label_dict = {idx: label for idx, label in enumerate(self.class_labels)}
            class_labels = np.array([label_dict[idx] for idx in predicted_indices])
            return class_labels
    
    def predict_proba(self, X):
        self.forward_propagation(X)
        return self.activations[-1]
        
    def forward_propagation(self, X):
        A = X
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._activation(Z)
            self.activations.append(A)

        Z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        A = self.last_layer_activation_function(Z)
        self.activations.append(A)

        return self.activations[-1]
        
    def backward_propagation(self, X, Y):
        m = X.shape[0]

        if self.class_type == 'classification':
            Y = self.one_hot_encode(labels=Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        dA = self.activations[-1] - Y 
        dZ = dA * self.last_layer_activation_derivative(A=self.activations[-1])

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

    def train(self, X_train, Y_train, validation_data, patience=5):
        self._init_weights()
        self.train_losses = []
        self.val_losses = []

        # project_name='"MLPClassifier_hyperparameter_tuning"'
        # if self.class_type == 'regression':
        #     project_name="MLPRegression_hyperparameter_tuning"

        # wandb.init(project=project_name, config={
        #     "learning_rate": self.learning_rate,
        #     "epochs": self.epochs,
        #     "batch_size": self.batch_size,
        #     "hidden_layers": self.hidden_layers,
        #     "activation_function": self.activation_function,
        #     "optimizer": self.optimizer,
        # })

        early_stopping_counter = 0
        best_loss = float('inf')
        for epoch in range(self.epochs):
            self._optimize(X_train, Y_train)

            if validation_data is not None:
                X_val, Y_val = validation_data
                val_loss = self.compute_loss(X_val, Y_val)
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     early_stopping_counter = 0
                # else:
                #     early_stopping_counter += 1
                #     if early_stopping_counter >= patience:
                #         print(f"Early stopping at epoch {epoch}")
                #         break

            train_loss = self.compute_loss(X_train, Y_train)
            self.train_losses.append(train_loss)

            if validation_data is not None:
                X_val, Y_val = validation_data
                val_loss = self.compute_loss(X_val, Y_val)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{self.epochs} complete, Train Loss : {train_loss}, Val Loss : {val_loss}")

            #     if self.class_type == 'regression':
            #         y_pred = self.predict(X_val)
            #         wandb.log({
            #             "train_loss": train_loss,
            #             "val_loss": val_loss,
            #             "MAE": self.mean_absolute_error(y_pred, Y_val),
            #             "RMSE": self.rmse(y_pred, Y_val),
            #             "R_squared": self.r_squared(y_pred, Y_val),
            #         })
            #     else:
            #         y_pred = self.predict(X_train)
            #         train_evaluation = model_evaluation(y_true=Y_train, y_pred=y_pred, classes_list=self.class_labels)
            #         y_pred = self.predict(X_val)
            #         val_evaluation = model_evaluation(y_true=Y_val, y_pred=y_pred, classes_list=self.class_labels)
            #         wandb.log({
            #             "train_loss": train_loss,
            #             "val_loss": val_loss,
            #             "train_accuracy": train_evaluation.accuracy_score(),
            #             "val_accuracy": val_evaluation.accuracy_score(),
            #             "precision": val_evaluation.precision_score("macro"),
            #             "recall": val_evaluation.recall_score("macro"),
            #             "f1_score": val_evaluation.f1_score("macro")
            #         })
            # else:
            #     print(f"Epoch {epoch+1}/{self.epochs} complete, Train Loss : {train_loss}")
            #     if self.class_type == 'regression':
            #         wandb.log({
            #             "train_loss": train_loss,
            #         })
            #     else:
            #         y_pred = self.predict(X_train)
            #         train_evaluation = model_evaluation(y_true=Y_train, y_pred=y_pred, classes_list=self.class_labels)
            #         wandb.log({
            #             "train_loss": train_loss,
            #             "train_accuracy": train_evaluation.accuracy_score(),
            #             })

        # wandb.finish()


    def one_hot_encode(self, labels):
        encoding = np.eye(len(self.class_labels))
        label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        label_indices = np.array([label_to_index[label] for label in labels])

        if encoding[label_indices].ndim == 1:
            encoding[label_indices] = encoding[label_indices].reshape(-1, 1)
        return encoding[label_indices]
    
    def compute_loss(self, X, Y):
        if self.class_type == 'classification':
            Y_pred_proba = self.predict_proba(X)
            return self.cross_entropy_loss(Y_pred_proba, Y)
        else:
            Y_pred = self.predict(X)
            if self.loss_function == 'bce':
                return self.binary_cross_entropy_loss(Y_pred, Y)
            else:
                return self.mean_squared_error(Y_pred, Y)
            
    def cross_entropy_loss(self, predictions, Y_true):
        Y_true_encoded = self.one_hot_encode(Y_true)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
        loss =  -np.mean(np.sum(Y_true_encoded * np.log(predictions), axis=1))
        return loss
    
    def binary_cross_entropy_loss(self, y_pred, y_true):
        y_true = y_true.reshape(-1, y_pred.shape[1])
        loss =  -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
        return loss

    def mean_squared_error(self, Y_pred, Y_true):
        Y_true = Y_true.reshape(-1, Y_pred.shape[1])
        error = np.mean((Y_pred - Y_true) ** 2)
        return error
    
    def mean_absolute_error(self, Y_pred, Y_true):
        Y_true = Y_true.reshape(-1, Y_pred.shape[1])
        error = np.mean(np.abs(Y_pred - Y_true))
        return error

    def rmse(self, Y_pred, Y_true):
        Y_true = Y_true.reshape(-1, Y_pred.shape[1])
        error = np.sqrt(np.mean((Y_pred - Y_true) ** 2))
        return error
    
    def r_squared(self, Y_pred, Y_true):
        Y_true = Y_true.reshape(-1, Y_pred.shape[1])
        Y_true_mean = np.mean(Y_true)
        total = np.sum((Y_true - Y_true_mean) ** 2)
        residual = np.sum((Y_true - Y_pred) ** 2)
        return 1 - (residual / total)

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

    def last_layer_activation_function(self, Z):
        if self.class_type == 'regression':
            if self.last_layer_activation == 'sigmoid':
               return self._sigmoid(Z)
            else:
               return self._linear(Z)
        else:
            return self.softmax(Z)
        
    def last_layer_activation_derivative(self, A):
        if self.class_type == 'regression':
            if self.last_layer_activation == 'sigmoid':
               return self._sigmoid_derivative(A)
            else:
               return self._linear_derivative(A)
        else:
            return self._linear_derivative(A)

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
        return (A > 0).astype(np.float64)
    
    def _linear_derivative(self, A):
        return np.ones_like(A)
    
    def softmax(self, Z):
        softmax_prob = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # stability fix
        softmax_prob /= np.sum(softmax_prob, axis=1, keepdims=True)
        return softmax_prob
    
    def _optimize(self, X_train, Y_train):
        if self.optimizer == 'sgd':
            self.stochastic_gradient_descent(X_train, Y_train)
        elif self.optimizer == 'batch-gd':
            self.batch_gradient_descent(X_train, Y_train)
        elif self.optimizer == 'mini-batch-gd':
            self.mini_batch_gradient_descent(X_train, Y_train)
        else:
            raise ValueError(f"Invalid optimizer {self.optimizer}")
        
    def _update_weights(self, grads_w, grads_b):
        for i in range(len(grads_w)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

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

    def gradient_check(self, X, Y, epsilon=1e-7):
        self.forward_propagation(X)
        grads_w, grads_b = self.backward_propagation(X, Y)

        for weight, bias, gradw, gradb in zip(self.weights, self.biases, grads_w, grads_b):
            num_gradw = np.zeros_like(weight)
            num_gradb = np.zeros_like(bias)

            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    original = weight[i, j]
                    weight[i, j] = original + epsilon
                    loss_plus_epsilon = self.compute_loss(X, Y)
                    weight[i, j] = original - epsilon
                    loss_minus_epsilon = self.compute_loss(X, Y)
                    weight[i, j] = original
                    num_gradw[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

            for i in range(bias.shape[1]):
                original = bias[0, i]
                bias[0, i] = original + epsilon
                loss_plus_epsilon = self.compute_loss(X, Y)
                bias[0, i] = original - epsilon
                loss_minus_epsilon = self.compute_loss(X, Y)
                bias[0, i] = original
                num_gradb[0, i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

            
            relative_difference = np.linalg.norm(gradw - num_gradw) / (np.linalg.norm(gradw + num_gradw) + 1e-8)
 
            if relative_difference > 10-6:
                print(f"Gradient check failed: relative difference is {relative_difference}")
                return False
            
            relative_difference = np.linalg.norm(gradb - num_gradb) / (np.linalg.norm(gradb + num_gradb) + 1e-8)

            if relative_difference > 10-6:
                print(f"Gradient check failed: relative difference is {relative_difference}")
                return False
       
        print("Gradient check passed!")
        return True
    

    
    