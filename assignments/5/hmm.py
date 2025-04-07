from common import *
from hmm_dataset import extract_mfcc

data_dir = '../../data/external/spoken_digit_dataset/recordings'
personal_records_dir = '../../data/external/personal_recordings'

## 3.3 Model Training

def train_hmm_models(data_by_digit, n_components=5, n_iter=100):
    models = {}
    for digit, mfcc_list in data_by_digit.items():
        lengths = [mfcc.shape[0] for mfcc in mfcc_list]
        X = np.vstack(mfcc_list)
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
        model.fit(X, lengths)
        models[digit] = model
        print(f"Training Completed For Digit - '{digit}'")
    return models

def predict_digit(models, mfcc_features):
    scores = {}
    for digit, model in models.items():
        try:
            score = model.score(mfcc_features)
            scores[digit] = score
        except:
            scores[digit] = -np.inf  
    prediction = max(scores, key=scores.get)
    return prediction

def evaluate_hmm_models(models, test_data_by_digit):
    correct_predictions = 0
    total_predictions = 0

    for digit, mfcc_list in test_data_by_digit.items():
        for mfcc_features in mfcc_list:
            predicted_digit = predict_digit(models, mfcc_features)
            if predicted_digit == digit:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

train_data_by_digit = {str(digit): [] for digit in range(10)}
test_data_by_digit = {str(digit): [] for digit in range(10)}

for file_name in os.listdir(data_dir):
    if file_name.endswith('.wav'):
        digit = file_name.split('_')[0]  
        file_path = os.path.join(data_dir, file_name)
        mfcc_features = extract_mfcc(file_path)
        mfcc_features = mfcc_features.T
        train_data_by_digit[digit].append(mfcc_features)

for digit, mfcc_list in train_data_by_digit.items():
    np.random.shuffle(mfcc_list)  
    split_index = int(len(mfcc_list) * 0.8)
    train_data_by_digit[digit] = mfcc_list[:split_index]
    test_data_by_digit[digit] = mfcc_list[split_index:]

models = train_hmm_models(train_data_by_digit, n_components=5, n_iter=100)
accuracy = evaluate_hmm_models(models, test_data_by_digit)
print(f"Model Accuracy on Test Set: {accuracy * 100:.4f}%")

personal_records = {str(digit): [] for digit in range(10)}
for file_name in os.listdir(personal_records_dir):
    if file_name.endswith('.wav'):
        digit = file_name.split('.')[0]  
        file_path = os.path.join(personal_records_dir, file_name)
        mfcc_features = extract_mfcc(file_path)
        mfcc_features = mfcc_features.T
        personal_records[digit].append(mfcc_features)

accuracy = evaluate_hmm_models(models, personal_records)
print(f"Model Accuracy on Personal Recordings: {accuracy * 100:.4f}%")


