from common import *
import numpy as np
import matplotlib.pyplot as plt
import time

def measure_inference_time(model, x_test) -> float:
    start_time = time.time()
    for x in x_test:
        x = x.reshape(1, -1)
        model.predict(x)
    end_time = time.time()
    return end_time - start_time

def plot_inference_time_for_different_models(x_train_red:np.ndarray, x_train:np.ndarray, y_train:np.ndarray, x_val:np.ndarray, x_val_red:np.ndarray) -> None:
    knn_complete_data = kNNClassifier(k=11,distance_method='manhattan')   
    knn_reduced_data = kNNClassifier(k=11,distance_method='manhattan')  

    knn_complete_data.fit(x_train, y_train)
    knn_reduced_data.fit(x_train_red, y_train)

    completed_data_time = measure_inference_time(knn_complete_data, x_val)
    reduced_data_time = measure_inference_time(knn_reduced_data, x_val_red)

    models = ['KNN on complete dataset', 'KNN on Reduced Dataset']
    inference_times = [completed_data_time, reduced_data_time]

    plt.figure(figsize=(8, 6))
    plt.bar(models, inference_times, color=['blue','orange'])

    plt.title('Inference Time of KNN models')
    plt.xlabel('KNN Model')
    plt.ylabel('Inference Time (seconds)')
    plt.savefig('figures/knn_models_inf_time.png')
    # plt.show()

#### ploting scree plot for spotify dataset
df = pd.read_csv('../../data/external/spotify.csv')
df = df.drop_duplicates(subset='track_id', keep='first')
df = df.drop(columns=['Unnamed: 0','track_id','artists','album_name','track_name'])
data = df.to_numpy()
np.random.shuffle(data)
features = data[:, :-1].astype(np.float64)
labels = data[:,-1]
classes_list = np.unique(labels)

norm_features = z_score_normalization(features)
# plot_scree_plot(norm_features, n_components=15, outPath='figures/spotify_scree_plot.png')  

# ### Dimensionality Reduction on Spotify Dataset
opt_dims = 13
pca = PrincipalComponentAnalysis(num_components=opt_dims)
pca.fit(norm_features)
transformed_features = pca.transform(norm_features)

train_ratio = 0.8
val_ratio = 0.2
train_size = int(train_ratio * len(data))
val_size = int(val_ratio * len(data))

x_train = transformed_features[:train_size]
y_train = labels[:train_size]

x_val = transformed_features[train_size:train_size + val_size]
y_val = labels[train_size:train_size + val_size]

k_best = 11
best_dist_metric = 'manhattan'
knn = kNNClassifier(k=k_best, distance_method=best_dist_metric)
knn.fit(x_train, y_train)
y_pred = np.array([knn.predict(x) for x in x_val])

evaluation = model_evaluation(y_true=y_val, y_pred=y_pred, classes_list=classes_list)
print(f'Accuracy : {evaluation.accuracy_score()}')
print(f'Precision (macro) : {evaluation.precision_score(method="macro")}')
print(f'Recall (macro): {evaluation.recall_score(method="macro")}')
print(f'F1-Score (macro): {evaluation.f1_score(method="macro")}')
print(f'Precision (micro) : {evaluation.precision_score(method="micro")}')
print(f'Recall (micro): {evaluation.recall_score(method="micro")}')
print(f'F1-Score (micro): {evaluation.f1_score(method="micro")}')


#### 
x_train_org = norm_features[:train_size]
x_val_org = norm_features[train_size:train_size + val_size]
# plot_inference_time_for_different_models(x_train=x_train_org, x_train_red=x_train, y_train=y_train, x_val=x_val_org, x_val_red=x_val)
