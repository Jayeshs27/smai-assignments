from common import *


best_k = 11
best_distance_metric = 'manhattan'

sp2_train_data = read_spotify_data('../../data/external/spotify-2/train.csv')
sp2_val_data = read_spotify_data('../../data/external/spotify-2/validate.csv')
sp2_test_data = read_spotify_data('../../data/external/spotify-2/test.csv')

classes_list = np.unique(sp2_train_data[:, -1])
classes_list = np.union1d(classes_list, np.unique(sp2_test_data[:, -1]))
classes_list = np.union1d(classes_list, np.unique(sp2_val_data[:, -1]))

sp2_x_train, sp2_y_train, sp2_x_val, sp2_y_val, sp2_x_test, sp2_y_test = preprocess_data(sp2_train_data, sp2_val_data, sp2_test_data)


knn = kNNClassifier(k=best_k, distance_method=best_distance_metric)
knn.fit(x_train=sp2_x_train, y_train=sp2_y_train)

sp2_y_pred_val = np.array([knn.predict(x) for x in sp2_x_val])
sp2_y_pred_test = np.array([knn.predict(x) for x in sp2_x_test])


print('3.2.1 : Spotify-2 Dataset Performance')
print('Train Dataset')
print_model_performance(y_true=sp2_y_val, y_pred=sp2_y_pred_val, classes_list=classes_list)
print('Test Dataset')
print_model_performance(y_true=sp2_y_test, y_pred=sp2_y_pred_test, classes_list=classes_list)
print("")
