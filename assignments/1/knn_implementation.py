from common import *


data = read_spotify_data('../../data/external/spotify.csv')
classes_list = np.unique(data[:,-1])
train_data, val_data, test_data = train_test_val_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(train_data, val_data, test_data)

k_val = 10
knn = kNNClassifier(k=k_val, distance_method='manhattan')
knn.fit(x_train=x_train, y_train=y_train)
y_pred = np.array([knn.predict(x) for x in x_val])

print(f'2.3.1 : KNN evalutaion scores(for k={k_val})')
print_model_performance(y_true=y_val, y_pred=y_pred, classes_list=classes_list)
print("")