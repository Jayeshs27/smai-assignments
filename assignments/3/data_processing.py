from common import *
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, MinMaxScaler
from sklearn.impute import SimpleImputer

# #### Data Processing - Wine Quality Dataset
df = pd.read_csv('../../data/external/WineQT.csv')
df = df.sample(frac=1).reset_index(drop=True)  
feature_labels = df.columns[:-1]
description = df[feature_labels].describe().T[['mean', 'std', 'min', 'max']]
print("Wine Quality Description:")
print(description)
plot_data_distribution(feature_labels=feature_labels, df=df, filePath='figures/WineQT_data_distribution.png')

quality = df['quality']
feature_labels = df.columns[:-2]
imputer = SimpleImputer(strategy='mean')
df[feature_labels] = imputer.fit_transform(df[feature_labels])

scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[feature_labels])
standardized_df = pd.DataFrame(standardized_data, columns=feature_labels)

normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(standardized_df)

normalized_df = pd.DataFrame(normalized_data, columns=feature_labels)
normalized_df['quality'] = quality.reset_index(drop=True)
output_file_path = '../../data/interim/3/WineQT_processed.csv'
normalized_df.to_csv(output_file_path, index=False)



# #### Data Processing - (2.6) Advertisement Dataset
df = pd.read_csv('../../data/external/advertisement.csv')
df = df.sample(frac=1).reset_index(drop=True)  
numerical_features = ['age', 'income', 'purchase_amount']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

categorical_features = ['education', 'occupation', 'most bought item']
df = pd.get_dummies(df, columns=categorical_features)

city_counts = df['city'].value_counts()
df['city'] = df['city'].map(city_counts)
df['gender'] = (df['gender'] == 'Male').astype(np.uint8)
df['labels_split'] = df['labels'].apply(lambda x: x.split())

mlb = MultiLabelBinarizer()
labels_one_hot = pd.DataFrame(mlb.fit_transform(df['labels_split']), columns=mlb.classes_)

df = df.drop(['labels', 'labels_split'], axis=1)
df = pd.concat([df, labels_one_hot], axis=1)

df.to_csv('../../data/interim/3/advertisement_processed.csv', index=False)


# #### Data Processing -(3) Boston Housing Dataset
df = pd.read_csv('../../data/external/HousingData.csv')
df = df.sample(frac=1).reset_index(drop=True)  
features_labels = df.columns
description = df.describe().T[['mean', 'std', 'min', 'max']]
print("Boston Housing Dataset Description:")
print(description)
plot_data_distribution(feature_labels=features_labels, df=df, filePath='figures/Housing_data_distribution.png')

medv = df['MEDV']
feature_labels = df.columns[:-1]
imputer = SimpleImputer(strategy='mean')
df[feature_labels] = imputer.fit_transform(df[feature_labels])

processed_df = df[feature_labels]
scaler = StandardScaler()
standardized_data = scaler.fit_transform(processed_df)
processed_df = pd.DataFrame(standardized_data, columns=feature_labels)

normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(processed_df)
processed_df = pd.DataFrame(normalized_data, columns=feature_labels)

processed_df['MEDV'] = medv.reset_index(drop=True)

output_file_path = '../../data/interim/3/HousingData_processed.csv'
processed_df.to_csv(output_file_path, index=False)



# #### Data Processing -(3.5) Diabetes Dataset
df = pd.read_csv('../../data/external/diabetes.csv')
df = df.sample(frac=1).reset_index(drop=True)  

labels = df['Outcome']
feature_labels = df.columns[:-1]
processed_df = df[feature_labels]

normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(processed_df[feature_labels])
processed_df = pd.DataFrame(normalized_data, columns=feature_labels)

processed_df['Outcome'] = labels.reset_index(drop=True)
processed_df.to_csv('../../data/interim/3/diabetes_processed.csv', index=False)



# #### Data Processing -(3) Spotify Dataset
df = pd.read_csv('../../data/external/spotify.csv')
df = df.drop_duplicates(subset='track_id', keep='first')
df = df.drop(columns=['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name'])
df = df.sample(frac=1).reset_index(drop=True)  

labels = df['track_genre']
feature_labels = df.columns[:-1]
processed_df = df[feature_labels]

binary_features = ['explicit', 'mode']  
numerical_features = [col for col in feature_labels if col not in binary_features]

normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(processed_df[numerical_features])
normalized_df = pd.DataFrame(normalized_data, columns=numerical_features)

normalized_df[binary_features] = df[binary_features].reset_index(drop=True)

normalized_df['track_genre'] = labels.reset_index(drop=True)
normalized_df.to_csv('../../data/interim/3/spotify_processed.csv', index=False)

