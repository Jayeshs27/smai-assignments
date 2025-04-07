from common import *

def plot_regression(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, y_pred:np.ndarray, 
                    mse_list:list, var_list:list, std_list:list, curr_iter:int, total_iter:int, degree:int, fileName:str) -> None:

    sorted_inds = np.argsort(x_train)
    x_train_sorted = x_train[sorted_inds]
    y_pred_sorted = y_pred[sorted_inds]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].scatter(x_train, y_train, label='Train Data', s=10, color='blue')
    axes[0, 0].scatter(x_test, y_test, label='Test Data', s=10, color='red')
    axes[0, 0].plot(x_train_sorted, y_pred_sorted, label='fitted line', color='orange',lw=2)
    axes[0, 0].axhline(0, color='black', linewidth=0.5)
    axes[0, 0].axvline(0, color='black', linewidth=0.5)
    axes[0, 0].set_xlabel('X-axis')
    axes[0, 0].set_ylabel('Y-axis')
    axes[0, 0].legend()
    

    iters = list(range(1, curr_iter + 1))
    axes[0, 1].plot(iters, mse_list, color='orange', lw=2)
    axes[0, 1].set_xlabel('Number of Iterations')
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].set_xlim(0, total_iter)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(iters, var_list, color='orange', lw=2)
    axes[1, 0].set_xlabel('Number of Iterations')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_xlim(0, total_iter)
    axes[1, 0].set_ylim(0, 2)

    axes[1, 1].plot(iters, std_list, color='orange', lw=2)
    axes[1, 1].set_xlabel('Number of Iterations')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_xlim(0, total_iter)
    axes[1, 1].set_ylim(0, 2)

    plt.suptitle(f'Regression For Degree {degree}')
    plt.tight_layout()
    plt.savefig(fileName)
    # plt.show()
    plt.close()


df = pd.read_csv('../../data/external/linreg.csv')
data = df.to_numpy()
train, val, test = train_test_val_split(data, train_ratio=0.8, test_ratio=0.2, val_ratio=0)
y_train = train[:,-1]
x_train = train[:, 0]
y_test = test[:,-1]
x_test = test[:, 0]

#################### Regression with degree = 1 ###########################

model = linearRegression(learning_rate=0.1)
total_iter = 1000
model.fit(x_train, y_train,total_iter)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)
print('3.1.1 : Simple Regression with Degree=1')
print_regression_performance(y_train=y_train, y_pred_train=y_pred_train, y_test=y_test, y_pred_test=y_pred_test)
print('')

plt.figure(figsize=(8,6))
plt.scatter(x_train, y_train, label='Train Data', s=15, color='blue')
plt.scatter(x_test, y_test, label='Test Data', s=15, color='red')
plt.plot(x_train, y_pred_train, label='fitted line', color='orange')
plt.title('Linear Regression for Degree = 1')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.savefig('figures/lr_degree_1.png')
# plt.show()

#################### Regression with degree > 1 ###########################
 

k = 10
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
model = regression(degree=k, learning_rate=0.1)
total_iter = 1000   
model.fit(x_train, y_train, total_iter)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)
print(f'3.1.2 : Regression with Degree={k}')
print_regression_performance(y_train=y_train, y_pred_train=y_pred_train, y_test=y_test, y_pred_test=y_pred_test)
print('')

plt.figure(figsize=(8,6))
plt.scatter(x_train, y_train, label='Train Data', s=15, color='blue')
plt.scatter(x_test, y_test, label='Test Data', s=15, color='red')
plt.plot(x_train, y_pred_train, label='fitted line', color='green')
plt.title('Linear Regression for Degree = 1')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.savefig(f'figures/regression_degree={k}.png')
# plt.show()


#######################  model on best k parameters #####################

best_k = 20
with open('best_k_parameters.txt', 'r') as file:
    lines = file.readlines()

parameters = [float(line.strip()) for line in lines]
model = regression(degree=best_k)
model.weights = np.array(parameters).reshape((len(parameters), 1))

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

print(f'Regression with best k value({best_k}):')
print_regression_performance(y_train=y_train, y_pred_train=y_pred_train, y_test=y_test, y_pred_test=y_pred_test)


##################  plotting curve fitting frames ####################

# k=20
# model = regression(degree=k)
# model.fit(x_train, y_train)
# model.weights = np.ones((k + 1, 1))  * 0.5

# mse_list=[]
# var_list=[]
# std_list=[]

# total_iter = 100
# for i in range(0, total_iter):
#     y_pred_train = model.predict(x_train)
#     grad = model.calculate_gradient()
#     model.weights -= lr * grad
    
#     mse_list.append(mean_squared_error(y_pred_train, y_train))
#     var_list.append(np.var(y_pred_train))
#     std_list.append(np.std(y_pred_train))

#     fileName=f'figures/reg_iter_{i + 1}.jpg'
#     plot_regression(x_train, y_train, x_test, y_test, 
#                     y_pred_train,
#                     mse_list,
#                     var_list,
#                     std_list,
#                     i + 1,
#                     total_iter,
#                     k,
#                     fileName)
