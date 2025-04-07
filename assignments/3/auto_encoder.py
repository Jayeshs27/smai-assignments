from common import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.read_csv('../../data/interim/3/spotify_processed.csv')
data = df.to_numpy()
X = data[:, :-1].astype(np.float64)
labels = data[:,-1]
classes_list = np.unique(labels)

# wandb.login()

sweep_config = {
    'method': 'grid',  
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },    
    'parameters': {
        'learning_rate': {
            'values': [0.01]
        },
        'activation_function': {
            'values': ['relu']
        },
        'batch_size': {
            'values': [32]
        },
        'optimizer': {
            'values': ['mini-batch-gd']
        },
        'encoder_hidden_layers': {
            'values': [[64,32]]
        },
        'epochs':{
            'values': [1000]
        }
    }
}

def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        autoencoder = AutoEncoder(
                          input_dim=15, 
                          reduced_dim=12, 
                          encoder_hidden_layers=config.encoder_hidden_layers,
                          decoder_hidden_layers=config.encoder_hidden_layers[::-1],
                          learning_rate=config.learning_rate, 
                          activation_function=config.activation_function, 
                          epochs=config.epochs, 
                          optimizer=config.optimizer,
                          batch_size=config.batch_size)

        autoencoder.fit(X)

# sweep_id = wandb.sweep(sweep_config, project="AutoEncoder_hyperparameter_tuning")
# wandb.agent(sweep_id, train_sweep)

best_model = AutoEncoder(input_dim=15, 
                          reduced_dim=12, 
                          encoder_hidden_layers=[64,32],
                          decoder_hidden_layers=[32,64],
                          learning_rate=0.01, 
                          activation_function='relu', 
                          epochs=1000, 
                          optimizer='mini-batch-gd',
                          batch_size=32)

best_model.fit(X=X)
reduced_data = best_model.get_latent(X=X)
print(X)
print(best_model.reconstruct(X))

train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(data))
val_size = int(val_ratio * len(data))

x_train = reduced_data[:train_size]
y_train = labels[:train_size]

x_val = reduced_data[train_size:train_size + val_size]
y_val = labels[train_size:train_size + val_size]


## KNN on reduced dataset
# k_best = 11
# best_dist_metric = 'manhattan'
# knn = kNNClassifier(k=k_best, distance_method=best_dist_metric)
# knn.fit(x_train, y_train)
# y_pred = np.array([knn.predict(x) for x in x_val])

# evaluation = model_evaluation(y_true=y_val, y_pred=y_pred, classes_list=classes_list)
# print(f'Accuracy : {evaluation.accuracy_score()}')
# print(f'Precision (macro) : {evaluation.precision_score(method="macro")}')
# print(f'Recall (macro): {evaluation.recall_score(method="macro")}')
# print(f'F1-Score (macro): {evaluation.f1_score(method="macro")}')
# print(f'Precision (micro) : {evaluation.precision_score(method="micro")}')
# print(f'Recall (micro): {evaluation.recall_score(method="micro")}')
# print(f'F1-Score (micro): {evaluation.f1_score(method="micro")}')


## MLP Classification on spotify dataset

mlp_model = MLP(input_size=12,     
                output_size=len(classes_list),
                hidden_layers=[32, 48], 
                learning_rate=0.001, 
                class_labels=classes_list,
                activation_function='tanh',
                optimizer='mini-batch-gd',   
                batch_size=16,    
                epochs=1500)   

mlp_model.fit(x_train, y_train, validation_data=(x_val, y_val))
y_pred = mlp_model.predict(x_val)
evaluation = model_evaluation(y_true=y_val, y_pred=y_pred, classes_list=classes_list)
print(f'Accuracy : {evaluation.accuracy_score()}')
print(f'Precision (macro) : {evaluation.precision_score(method="macro")}')
print(f'Recall (macro): {evaluation.recall_score(method="macro")}')
print(f'F1-Score (macro): {evaluation.f1_score(method="macro")}')
print(f'Precision (micro) : {evaluation.precision_score(method="micro")}')
print(f'Recall (micro): {evaluation.recall_score(method="micro")}')
print(f'F1-Score (micro): {evaluation.f1_score(method="micro")}')










