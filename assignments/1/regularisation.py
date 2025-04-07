from common import *

df = pd.read_csv('../../data/external/regularisation.csv')

data = df.to_numpy()
train, val, test = train_test_val_split(data, 0.8, 0.2, 0)

y_train = train[:,-1]
y_train = y_train.reshape(-1, 1)
x_train = train[:, 0]

y_test = test[:,-1]
y_test = y_test.reshape(-1, 1)
x_test = test[:, 0]

opt_k=1
lr=0.15
num_iter = 500
k_vales = [15]

print('3.2.1 : Regularization')
for reg_type in [None,'L1','L2']:
    for k in k_vales:

        print(f'For k={k}, Reg_type={reg_type}: ')
        model = regression(degree=k, regularization_type=reg_type, regularization_parm=0.09)
        model.fit(x_train, y_train)
        model.weights = np.ones((k + 1, 1)) * 1

        for j in range(num_iter):
            gradients = model.calculate_gradient()
            model.weights -= lr * gradients

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        print_regression_performance(y_train, y_pred_train, y_test, y_pred_test)

        # plt.figure(figsize=(8,6))
        # sorted_inds = np.argsort(x_train)
        # x_train_sorted = x_train[sorted_inds]
        # y_pred_sorted = y_pred_train[sorted_inds]

        # plt.scatter(x_train, y_train, label='Train Data', s=15, color='blue')
        # plt.scatter(x_test, y_test, label='Test Data', s=15, color='red')
        # plt.plot(x_train_sorted, y_pred_sorted, label='fitted line', color='orange')
        # plt.title('Linear Regression for Degree = 1')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.axhline(0, color='black', linewidth=0.5)
        # plt.axvline(0, color='black', linewidth=0.5)
        # plt.legend()
        # plt.savefig(f'figures/3_2_k={k}_regul_type={reg_type}.png')
