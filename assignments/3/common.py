
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../performance-measures")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/MLP")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/AutoEncoders")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/knn")))

from evaluation import model_evaluation
from MLP import MLP
from AutoEncoders import AutoEncoder
from knn import kNNClassifier


def train_test_val_split(data:np.ndarray, train_ratio:float, val_ratio:float) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.shuffle(data)
    train_size = int(train_ratio * len(data))
    val_size = int(val_ratio * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    x_val = val_data[:, :-1]
    y_val = val_data[:, -1]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    return (x_train, y_train, x_val, y_val, x_test, y_test)

def plot_data_distribution(feature_labels:list, df:pd.DataFrame, filePath:str):
    plt.figure(figsize=(15, 10)) 
    for i, column in enumerate(feature_labels, 1):
        plt.subplot(len(feature_labels) // 4 + 1, 4, i)  
        plt.hist(df[column], bins=40, color='skyblue', edgecolor='black')  
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()  
    plt.savefig(filePath)
    # plt.show()

def z_score_normalization(data:np.ndarray) -> np.ndarray:
    std_dev = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    std_dev = np.where(std_dev < 1e-6, 1e-6, std_dev)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def plot_losses(train_losses, val_losses, loss_type, out_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_type} Loss')
    plt.title(f'{loss_type} Loss vs Epochs')
    plt.legend()
    plt.savefig(out_path)
    plt.show()