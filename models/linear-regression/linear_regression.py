import numpy as np

class linearRegression:
    def __init__(self, learning_rate:float=0.5):
        self.learning_rate = learning_rate
        self.x_points = np.empty((0,0))
        self.y_points = np.empty((0,0))
        self.weight=0.1
        self.bias=0.1

    def fit(self, x_points:np.ndarray, y_points:np.ndarray, iterations:int) -> None:
        self.x_points = x_points
        self.y_points = y_points
        for _ in range(iterations):
            b_grad, w_grad = self.calculate_gradient()
            self.bias -= self.learning_rate * b_grad
            self.weight -= self.learning_rate * w_grad
            print(self.bias, self.weight)

    def predict(self, x_points:np.ndarray) -> np.ndarray:
        y_pred = x_points * self.weight + self.bias
        return y_pred
    
    def calculate_gradient(self) -> tuple[float, float]:
        errors = self.y_points - (self.weight * self.x_points + self.bias)
        bias_grad = - 2 * np.mean(errors)
        weight_grad = - 2 * np.mean(self.x_points * errors)
        return bias_grad, weight_grad
    
class regression:
    
    def __init__(self, degree=1, regularization_type=None, regularization_parm=0.01, learning_rate=0.1):
        self.degree = degree
        self.x_poly_feat = None
        self.y_true = None
        self.weights = None
        if regularization_type not in [None, 'L1', 'L2']:
            print('Invalid regularization type')
            exit(1)
        self.learning_rate = learning_rate
        self.regularization_type = regularization_type
        self.regularization_parm = regularization_parm

    def fit(self, x_points:np.ndarray, y_points:np.ndarray, iterations:int) -> None:
        self.x_poly_feat = np.vander(x_points, self.degree + 1, increasing=True)
        self.y_true = y_points
        self.weights = np.ones((self.degree + 1, 1)) * 0.5
        for i in range(iterations):
            grad = self.calculate_gradient()
            self.weights -= self.learning_rate * grad
        
    def predict(self, x_points:np.ndarray) -> np.ndarray:
        poly_features = np.vander(x_points, self.degree + 1, increasing=True)
        y_pred = poly_features @ self.weights
        return y_pred

    def calculate_gradient(self) -> np.ndarray:
        size = len(self.y_true)
        y_pred = self.x_poly_feat @ self.weights

        error = y_pred - self.y_true.reshape(-1,1) 
        gradient = (2 * self.x_poly_feat.T @ error) / size

        if self.regularization_type == 'L1':
            gradient += self.regularization_parm * np.sum(np.sign(self.weights))
        elif self.regularization_type == 'L2':
            gradient += 2 * self.regularization_parm * np.sum(self.weights)

        return gradient