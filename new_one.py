import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the configuration from the YAML file
with open('pipeline_config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Generate a synthetic dataset
np.random.seed(0)
n_samples = 1000
n_features = 10

# Create numerical features
X_num = np.random.rand(n_samples, n_features)

# Create categorical features
X_cat = np.random.choice(['A', 'B', 'C'], size=(n_samples, 3))

# Combine numerical and categorical features
X = np.hstack((X_num, X_cat))

# Generate binary labels
y = np.random.randint(0, 2, n_samples)

# Create a DataFrame
columns = [f'num_feature_{i}' for i in range(n_features)] + ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']
df = pd.DataFrame(X, columns=columns)
df['label'] = y

# Introduce some missing values
nan_mask = np.random.rand(*df.shape) < 0.1
df = df.mask(nan_mask)

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the numerical and categorical columns
num_features = [f'num_feature_{i}' for i in range(n_features)]
cat_features = ['cat_feature_1', 'cat_feature_2', 'cat_feature_3']

# Create the preprocessing pipeline for numerical features
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=config['preprocessing']['numerical']['imputer_strategy'])),  # Impute missing values
    ('scaler', StandardScaler())  # Scale features
])

# Create the preprocessing pipeline for categorical features
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=config['preprocessing']['categorical']['imputer_strategy'])),  # Impute missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Create the full pipeline with feature selection and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing
    ('feature_selection', SelectKBest(score_func=f_classif, k=config['feature_selection']['k'])),  # Feature selection
    ('classifier', RandomForestClassifier(**config['classifier']['params']))  # Classifier
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
