from common import *


def plot_effect_of_activation(best_model):
    activation_functions = ['sigmoid','relu','tanh','linear']
    train_losses_list = []
    val_losses_list = []
    for activation in activation_functions:
        best_model.activation_function = activation
        best_model.fit(x_train, y_train, validation_data=(x_val, y_val))
        train_losses_list.append(best_model.train_losses)
        val_losses_list.append(best_model.val_losses)

    plt.figure(figsize=(10, 6))
    for i, activation in enumerate(activation_functions):
        epochs = range(1, len(train_losses_list[i]) + 1)
        plt.plot(epochs, train_losses_list[i], label=f'Train Loss ({activation})')
        plt.plot(epochs, val_losses_list[i], '--', label=f'Val Loss ({activation})')

    plt.title('Effect of Activation Functions on Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/2_5_effect_of_activation.png')
    # plt.show()

def plot_effect_of_learning_rate(best_model):
    learning_rates = [0.001, 0.005, 0.01, 0.1]
    train_losses_list = []
    val_losses_list = []
    for lr in learning_rates:
        best_model.learning_rate = lr
        best_model.fit(x_train, y_train, validation_data=(x_val, y_val))
        train_losses_list.append(best_model.train_losses)
        val_losses_list.append(best_model.val_losses)

    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(learning_rates):
        epochs = range(1, len(train_losses_list[i]) + 1)
        plt.plot(epochs, train_losses_list[i], label=f'Train Loss (lr={lr})')
        plt.plot(epochs, val_losses_list[i], '--', label=f'Val Loss (lr={lr})')

    plt.title('Effect of Learnning Rate on Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/2_5_effect_of_learning_rate.png')
    # plt.show()

def plot_effect_of_batch_size(best_model):
    batch_sizes = [16, 32, 128, 256]
    train_losses_list = []
    val_losses_list = []
    for bs in batch_sizes:
        best_model.optimizer = 'mini-batch-gd'
        best_model.batch_size = bs
        best_model.fit(x_train, y_train, validation_data=(x_val, y_val))
        train_losses_list.append(best_model.train_losses)
        val_losses_list.append(best_model.val_losses)

    plt.figure(figsize=(10, 6))
    for i, bs in enumerate(batch_sizes):
        epochs = range(1, len(train_losses_list[i]) + 1)
        plt.plot(epochs, train_losses_list[i], label=f'Train Loss (batch size={bs})')
        plt.plot(epochs, val_losses_list[i], '--', label=f'Val Loss (batch size={bs})')

    plt.title('Effect of Batch Size Rate on Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/2_5_effect_of_batch_size.png')
    # plt.show()


df = pd.read_csv('../../data/interim/3/WineQT_processed.csv')
data = df.to_numpy()
classes_list = np.unique(data[:,-1])
x_train, y_train, x_val, y_val, x_test, y_test = train_test_val_split(data=data, train_ratio=0.8, val_ratio=0.1)


sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },    
    'parameters': {
        'learning_rate': {
            'values': [0.008, 0.01]
        },
        'activation_function': {
            'values': ['tanh','relu','sigmoid','linear']
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
        mlp_model = MLP(input_size=11,     
                        output_size=6,     
                        hidden_layers=config.hidden_layers, 
                        class_labels=classes_list,
                        learning_rate=config.learning_rate, 
                        activation_function=config.activation_function,
                        optimizer=config.optimizer,   
                        batch_size=config.batch_size,    
                        epochs=config.epochs)

        mlp_model.fit(x_train, y_train, validation_data=(x_val, y_val))


# sweep_id = wandb.sweep(sweep_config, project="MLPClassifier_hyperparameter_tuning")
# wandb.agent(sweep_id, train_sweep)


best_model = MLP(input_size=11,     
                    output_size=6,     
                    hidden_layers=[64, 16], 
                    class_labels=classes_list,
                    learning_rate=0.005, 
                    activation_function='tanh',
                    optimizer='mini-batch-gd',   
                    batch_size=32,    
                    epochs=1500)   

### Performance on test set
best_model.fit(X_train=x_train, Y_train=y_train, validation_data=(x_val, y_val))
best_model.gradient_check(x_train, y_train, epsilon=1e-7)

y_pred = best_model.predict(x_test)
evaluation = model_evaluation(y_true=y_test, y_pred=y_pred, classes_list=classes_list)
print(f'Accuracy : {evaluation.accuracy_score()}')
print(f'Precision (macro) : {evaluation.precision_score(method="macro")}')
print(f'Recall (macro): {evaluation.recall_score(method="macro")}')
print(f'F1-Score (macro): {evaluation.f1_score(method="macro")}')
print(f'Precision (micro) : {evaluation.precision_score(method="micro")}')
print(f'Recall (micro): {evaluation.recall_score(method="micro")}')
print(f'F1-Score (micro): {evaluation.f1_score(method="micro")}')

## counting correct predictions
count_pred = {}
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        if y_test[i] not in count_pred:
            count_pred[y_test[i]] = 0
        count_pred[y_test[i]] += 1
for key, value in count_pred.items():
    print(key, ":", value)


# ### Analyzing Hyperparameters EFFects
plot_effect_of_activation(best_model=best_model)
plot_effect_of_learning_rate(best_model=best_model)
plot_effect_of_batch_size(best_model=best_model)
