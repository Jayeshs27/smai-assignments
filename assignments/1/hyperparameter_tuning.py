from common import *


def plot_k_vs_accuracy(kvals, accurs):
    plt.figure(figsize=(10,10))
    plt.figure(figsize=(10, 6))
    plt.plot(kvals, accurs, marker='o', linestyle='-')
    plt.title('KNN: Varying Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(kvals)
    plt.savefig('figures/k_vs_accuracy.png')
    # plt.show()

def print_best_k_and_distance_metric_pairs():
    data = read_spotify_data('../../data/external/spotify.csv')
    classes_list = np.unique(data[:,-1])
    train_data, val_data, test_data = train_test_val_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(train_data, val_data, test_data)

    accuracy_list=[]
    for k_val in range(1, 30):
        for method in ["manhattan"]:
            knn = kNNClassifier(k=k_val,distance_method=method)
            knn.fit(x_train=x_train, y_train=y_train)

            y_pred = np.array([])
            for x,y in zip(x_val, y_val):
                pred = knn.predict(x)
                y_pred = np.append(y_pred, pred)

            eval = model_evaluation(y_true=y_val, y_pred=y_pred, classes_list=classes_list)
            print(f"k : {k_val}, method : {method}, Accuracy: ", eval.accuracy_score())
            accur = eval.accuracy_score()
            accuracy_list.append([k_val, method, accur])
    
    return accuracy_list
    

print('2.4 : Hyperparamter Tuning')
accuracy_list = print_best_k_and_distance_metric_pairs()
sorted_list = sorted(accuracy_list, key=lambda x: x[2], reverse=True)

print("Best {K, Distance_method}")
size = min(11, len(sorted_list) + 1)
for i in range(size):
    print(sorted_list[i])

k_vals=[]
accuracy=[]
for item in accuracy_list:
    if item[2] == 'manhattan':
        k_vals.append(item[0])
        accuracy.append(item[2])

plot_k_vs_accuracy(k_vals, accuracy)
