from common import *
from MultiLabelMLP import MultiLabelMLP
from multi_label_metrices import *

df = pd.read_csv('../../data/interim/3/advertisement_processed.csv')
df_shuffled = df.sample(frac=1).reset_index(drop=True)
num_classes = 8
X = df.iloc[:, :-num_classes].values.astype(np.float64)
Y = df.iloc[:, -num_classes:].values.astype(np.float64)

train_ratio = 0.8
val_ratio = 0.1
train_size = int(train_ratio * len(X))
val_size = int(val_ratio * len(X))

x_train = X[:train_size]
y_train = Y[:train_size]
y_val = Y[train_size:train_size + val_size]
x_val = X[train_size:train_size + val_size]
x_test = X[train_size + val_size:]
y_test = Y[train_size + val_size:]

def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        mlp_model = MultiLabelMLP(input_size=46,     
                                    output_size=8,   
                                    class_labels=np.eye(N=num_classes),
                                    hidden_layers=config.hidden_layers,   
                                    learning_rate=config.learning_rate, 
                                    activation_function=config.activation_function,
                                    optimizer=config.optimizer,   
                                    batch_size=config.batch_size,    
                                    epochs=config.epochs)

        mlp_model.fit(x_train, y_train, validation_data=(x_val, y_val))


wandb.login()

sweep_config = {
    'method': 'grid',  
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },    
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.01]
        },
        'activation_function': {
            'values': ['tanh', 'sigmoid', 'relu', 'linear']
        },
        'batch_size': {
            'values': [16, 64]
        },
        'optimizer': {
            'values': ['mini-batch-gd']
        },
        'hidden_layers': {
            'values': [[64, 32]]
        },
        'epochs':{
            'values':[1500],
        }
    }
}

# sweep_id = wandb.sweep(sweep_config, project="MLPMultiLabel_hyperparameter_tuning")
# wandb.agent(sweep_id, train_sweep)

best_model = MultiLabelMLP(input_size=46, 
                          hidden_layers=[64,32], 
                          output_size=8, 
                          learning_rate=0.001,
                          class_labels=np.eye(N=num_classes),
                          activation_function='tanh',
                          optimizer='mini-batch-gd',
                          batch_size=16,
                          epochs=800
                          )

best_model.fit(X_train=x_train, Y_train=y_train, validation_data=(x_val, y_val))
best_model.gradient_check(x_train, y_train, epsilon=1e-7)

y_pred = best_model.predict(X=x_test)
evaluation = MultiLabelMetrics(y_true=y_test, y_pred=y_pred)
print(f'Accuracy : {evaluation.accuracy_score()}')
print(f'Precision (macro) : {evaluation.precision_score(method="macro")}')
print(f'Recall (macro): {evaluation.recall_score(method="macro")}')
print(f'F1-Score (macro): {evaluation.f1_score(method="macro")}')
print(f'Precision (micro) : {evaluation.precision_score(method="micro")}')
print(f'Recall (micro): {evaluation.recall_score(method="micro")}')
print(f'F1-Score (micro): {evaluation.f1_score(method="micro")}')
print(f'Hamming Loss : {evaluation.hamming_loss()}')
