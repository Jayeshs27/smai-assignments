import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/knn")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/linear-regression")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../performance-measures")))

from knn import kNNClassifier
from evaluation import model_evaluation
from linear_regression import linearRegression, regression

def z_score_normalization(data:np.ndarray, mean:np.ndarray, std_dev:np.ndarray) -> np.ndarray:
    std_dev[std_dev == 0] = 1
    return ((data - mean) / std_dev)

def mean_squared_error(y_pred:np.ndarray, y_true:np.ndarray) -> float:
    square_error = (y_true - y_pred) ** 2
    mse = np.mean(square_error)
    return mse

def train_test_val_split(data:np.ndarray, train_ratio:float, test_ratio:float, 
                     val_ratio:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.shuffle(data)
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return (train_data, val_data, test_data)


def read_spotify_data(filePath:str) -> tuple[pd.DataFrame]:   
    df = pd.read_csv(filePath)
    df = df.drop_duplicates(subset='track_id', keep='first')
    df = df.drop(columns=['Unnamed: 0','track_id','artists','album_name','track_name'])
    data = df.drop(columns=['explicit','time_signature'])
    data = data.to_numpy()
    return data

def preprocess_data(train_data:np.ndarray, val_data:np.ndarray, test_data:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                                               np.ndarray, np.ndarray, np.ndarray]:
    x_train = train_data[:, :-1]
    y_train = train_data[:,-1]
    x_train = x_train.astype(np.float64)
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = z_score_normalization(x_train, x_train_mean, x_train_std)

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    x_test = x_test.astype(np.float64)
    x_test = z_score_normalization(x_test, x_train_mean, x_train_std)

    x_val = val_data[:, :-1]
    y_val = val_data[:, -1]
    x_val = x_val.astype(np.float64)
    x_val = z_score_normalization(x_val, x_train_mean, x_train_std)

    return (x_train, y_train, x_val, y_val, x_test, y_test)

def print_model_performance(y_true:np.ndarray, y_pred:np.ndarray, classes_list:list):
    eval = model_evaluation(y_true, y_pred, classes_list)
    print(f'Accuracy : {eval.accuracy_score()}')
    print(f'Precision (macro) : {eval.precision_score(method="macro")}')
    print(f'Recall (macro): {eval.recall_score(method="macro")}')
    print(f'F1-Score (macro): {eval.f1_score(method="macro")}')
    print(f'Precision (micro) : {eval.precision_score(method="micro")}')
    print(f'Recall (micro): {eval.recall_score(method="micro")}')
    print(f'F1-Score (micro): {eval.f1_score(method="micro")}')

def print_regression_performance(y_train, y_pred_train, y_test, y_pred_test):
    mse_train = mean_squared_error(y_pred_train, y_train)
    mse_test = mean_squared_error(y_pred_test, y_test)
    print('For Train Data : ')  
    print(f'\tMSE : {mse_train}')
    print(f'\tVariance : {np.var(y_pred_train)}')
    print(f'\tStd Dev : {np.std(y_pred_train)}')
    print('For Test Data : ')  
    print(f'\tMSE : {mse_test}')
    print(f'\tVariance : {np.var(y_pred_test)}')
    print(f'\tStd Dev : {np.std(y_pred_test)}')
    print('------------------------------------------')