from common import *
from sklearn.neighbors import KNeighborsClassifier


def measure_inference_time(model, x_test) -> float:
    start_time = time.time()
    for x in x_test:
        x = x.reshape(1, -1)
        model.predict(x)
    end_time = time.time()
    return end_time - start_time

def plot_inference_time_for_different_models():
    data = read_spotify_data('../../data/external/spotify.csv')
    classes_list = np.unique(data[:,-1])
    train_data, val_data, test_data = train_test_val_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(train_data, val_data, test_data)

    best_knn = kNNClassifier(k=11,distance_method='manhattan')   
    optimized_knn = kNNClassifier()  
    default_knn = KNeighborsClassifier()

    default_knn.fit(x_train, y_train)
    best_knn.fit(x_train, y_train)
    optimized_knn.fit(x_train, y_train)

    default_time = measure_inference_time(default_knn, x_test)
    best_time = measure_inference_time(best_knn, x_test)
    optimized_time = measure_inference_time(optimized_knn, x_test)

    print(f'Inference Time KNN models :')
    print(f" Best KNN model inference time: {best_time:.6f} seconds")
    print(f" Optimized KNN model inference time: {optimized_time:.6f} seconds")
    print(f" Default sklearn KNN model inference time: {default_time:.6f} seconds")

    models = ['Best KNN', 'Optimized KNN', 'Default sklearn KNN']
    inference_times = [best_time, optimized_time, default_time]

    plt.figure(figsize=(8, 6))
    plt.bar(models, inference_times, color=['blue', 'green', 'orange'])

    plt.title('Inference Time for Different KNN Models')
    plt.xlabel('KNN Model')
    plt.ylabel('Inference Time (seconds)')
    plt.savefig('figures/knn_models_inference_time.png')
    # plt.show()

def plot_inference_time_vs_model_size():
    size_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    data = read_spotify_data('../../data/external/spotify.csv')
    classes_list = np.unique(data[:,-1])
    train_data, val_data, test_data = train_test_val_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(train_data, val_data, test_data)
    org_size = len(x_train)

    best_knn_inf_times=[]
    optimized_knn_inf_times=[]
    default_knn_inf_times=[]
    train_sizes = []

    best_knn = kNNClassifier(k=11,distance_method='manhattan')   
    optimized_knn = kNNClassifier()  
    default_knn = KNeighborsClassifier()

    for ratio in size_ratios:
        train_sizes.append(int(ratio * org_size))

        x_train_red_size = x_train[:int(ratio * org_size)]
        y_train_red_size = y_train[:int(ratio * org_size)]

        default_knn.fit(x_train_red_size, y_train_red_size)
        best_knn.fit(x_train_red_size, y_train_red_size)
        optimized_knn.fit(x_train_red_size, y_train_red_size)
        
        default_time = measure_inference_time(default_knn, x_test)
        best_time = measure_inference_time(best_knn, x_test)
        optimized_time = measure_inference_time(optimized_knn, x_test)

        best_knn_inf_times.append(best_time)
        optimized_knn_inf_times.append(optimized_time)
        default_knn_inf_times.append(default_time)

    plt.figure(figsize=(12, 8))

    plt.plot(train_sizes, best_knn_inf_times, color='red', label='Best kNN model')
    plt.plot(train_sizes, optimized_knn_inf_times, color='blue', label='Optimized kNN model')
    plt.plot(train_sizes, default_knn_inf_times, color='green', label='Default sklearn kNN model')

    plt.title('Inference Time vs Training Dataset Size')
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Inference Time (seconds)')
    plt.legend()
    plt.savefig('figures/knn_models_vs_data_size.png')
    # plt.show()

print('2.5.1 : Optimization')
plot_inference_time_for_different_models()
plot_inference_time_vs_model_size()
print('')