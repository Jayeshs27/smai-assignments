from common import *
    
df = pd.read_csv('../../data/interim/3/HousingData_processed.csv')
data = df.to_numpy()
x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(data=data, train_ratio=0.8, val_ratio=0.1)

# wandb.login()

sweep_config = {
    'method': 'grid',  
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },    
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.005]
        },
        'activation_function': {
            'values': ['relu','sigmoid','linear','tanh']
        },
        'batch_size': {
            'values': [32]
        },
        'optimizer': {
            'values': ['batch-gd']
        },
        'hidden_layers': {
            'values': [[64,16]]
        },
        'epochs':{
            'values': [2500]
        }
    }
}

def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        mlp_model = MLP(input_size=13,
                        output_size=1,     
                        class_type='regression',
                        hidden_layers=config.hidden_layers, 
                        learning_rate=config.learning_rate, 
                        activation_function=config.activation_function,
                        optimizer=config.optimizer,   
                        batch_size=config.batch_size,    
                        epochs=config.epochs)   

        mlp_model.fit(x_train, y_train, validation_data=(x_val, y_val))

# sweep_id = wandb.sweep(sweep_config, project="MLPRegression_hyperparameter_tuning")
# wandb.agent(sweep_id, train_sweep)

# Model Evaluation
best_model = MLP(input_size=13,   
                    class_type='regression', 
                    output_size=1,     
                    hidden_layers=[64,16], 
                    learning_rate=0.005, 
                    activation_function='relu',
                    optimizer='mini-batch-gd',   
                    batch_size=128,    
                    epochs=1500)   

# ### Performance on test set
best_model.fit(X_train=x_train, Y_train=y_train, validation_data=(x_val, y_val))
best_model.gradient_check(x_train, y_train, epsilon=1e-7)

y_pred = best_model.predict(x_test)
print(f'MSE : {best_model.mean_squared_error(Y_pred=y_pred, Y_true=y_test)}')
print(f'MAE : {best_model.mean_absolute_error(Y_pred=y_pred, Y_true=y_test)}')
print(f'RMSE : {best_model.rmse(Y_pred=y_pred, Y_true=y_test)}')
print(f'R-Squared : {best_model.r_squared(Y_pred=y_pred, Y_true=y_test)}')

## ploting MSE for each point
y_test = y_test.reshape(-1, 1)
squared_errors = (y_test - y_pred) ** 2
plt.figure(figsize=(10, 6))
plt.scatter(y_test, squared_errors, color='blue', alpha=0.6)
plt.title('MSE vs y_test values')
plt.xlabel('y_test values')
plt.ylabel('MSE (squared errors)')
plt.grid(True)
plt.savefig('figures/mse_per_point_reg.png')
# plt.show()


# 3.5 MSE v/s BCE on Diabetes Dataset
df = pd.read_csv('../../data/interim/3/diabetes_processed.csv')
data = df.to_numpy()
x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(data=data, train_ratio=0.8, val_ratio=0.1)

mlp_mse = MLP(input_size=8,
                output_size=1,  
                class_type='regression',  
                hidden_layers=[12, 6], 
                learning_rate=0.001, 
                activation_function='relu',
                optimizer='mini-batch-gd',   
                batch_size=8,    
                epochs=1000,
                last_layer_activation='sigmoid',
                loss_function='mse')

mlp_mse.fit(x_train, y_train, validation_data=(x_val, y_val))
mlp_mse.gradient_check(x_train, y_train)
plot_losses(mlp_mse.train_losses, mlp_mse.val_losses, loss_type='MSE', out_path='figures/3_5_mse_vs_epochs.png')

mlp_bce = MLP(input_size=8,
                output_size=1,     
                class_type='regression',
                hidden_layers=[12, 6], 
                learning_rate=0.01, 
                activation_function='relu',
                optimizer='mini-batch-gd',   
                batch_size=8,    
                epochs=1000,
                last_layer_activation='sigmoid',
                loss_function='bce')

mlp_bce.fit(x_train, y_train, validation_data=(x_val, y_val))
plot_losses(mlp_bce.train_losses, mlp_bce.val_losses, loss_type='BCE', out_path='figures/3_5_bce_vs_epochs.png')



