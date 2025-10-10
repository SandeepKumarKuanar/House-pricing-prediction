#### loading the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import RandomizedSearchCV

#### loading the dataset
df = pd.read_csv('dataset/kc_final.csv')
# print(df.head())
# print(df.info())

#### checking for inconsistencies in dataset
### checking for missing values
# print(df.isnull().sum()) ## no missing values

### checking for duplicate values
# print(df.duplicated().sum()) ## no duplicate values

### splitting the dataset into training and testing sets
X = df.drop(columns='price', axis=1)
columns_to_drop = ['id', 'date', 'Unnamed: 0'] ## they are just numbers, we need to focus on features that impact the pricing of the house
X = X.drop(columns=columns_to_drop, axis=1)
# print(X)
Y = df['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
# print(X) ## seperated features from pricing of the house
# print(Y) ## pricing of the house as one column

### joining the training data to check the distribution of the target variable
training_data = X_train.join(Y_train)
# plt.figure(figsize=(10, 6))
# sns.heatmap(training_data.corr(), annot=True, cmap='YlGnBu')

### taking log of the target variable to reduce the skewness, and make them look more like a normal distribution
training_data['bedrooms'] = np.log(training_data['bedrooms'] + 1)
training_data['bathrooms'] = np.log(training_data['bathrooms'] + 1)
training_data['sqft_living'] = np.log(training_data['sqft_living'] + 1)
training_data['floors'] = np.log(training_data['floors'] + 1)
training_data['sqft_basement'] = np.log(training_data['sqft_basement'] + 1)
training_data['yr_built'] = np.log(training_data['yr_built'] + 1)

### waterfront == proximity to the water, like ocean, river, lake, etc. 
### this are already in binary format
# print(training_data['waterfront'].value_counts()) ## 0 -> no waterfront, 1 -> waterfront

### visualizing the latitude and longitude of the houses
# training_data.hist(figsize=(10, 8))
# sns.scatterplot(data=training_data, x='lat', y='long', hue='price', palette='coolwarm')
# plt.tight_layout()
# plt.show()

#### feature engineering was a success below
### living quality would be feature engineering from view, condition, and grade
training_data['living_quality'] = training_data['view'] + (training_data['grade'] / 2) + training_data['condition']
training_data = training_data.drop(columns=['view', 'condition', 'grade'], axis=1)
# sns.heatmap(training_data.corr(), annot=True, cmap='YlGnBu')
# plt.tight_layout()
# plt.show()
# print(training_data.info())

## undoing the mess
X_train = training_data.drop(columns=['price'])
Y_train = training_data['price']

### I will go with the default random forest regressor model, without hyperparameter tuning, as it won't perform well on small datasets
forest = RandomForestRegressor()
# training the model
forest.fit(X_train, Y_train)
# print("The score on the training dataset:\n", forest.score(X_train, Y_train))

## testing the model 
X_test['living_quality'] = X_test['view'] + (X_test['grade'] / 2) + X_test['condition']
X_test = X_test.drop(columns=['view', 'condition', 'grade'])
X_test['bedrooms'] = np.log(X_test['bedrooms'] + 1)
X_test['bathrooms'] = np.log(X_test['bathrooms'] + 1)
X_test['sqft_living'] = np.log(X_test['sqft_living'] + 1)
X_test['floors'] = np.log(X_test['floors'] + 1)
X_test['sqft_basement'] = np.log(X_test['sqft_basement'] + 1)
X_test['yr_built'] = np.log(X_test['yr_built'] + 1)
# print("The score on the test dataset:\n", forest.score(X_test, Y_test))

### i got this classical overfitting problem, where the model performs well on training data but poorly on test data. . . as a 0.1 difference is quite high. . .as 
# --- Ran at 20:47:11
# The score on the training dataset:
#  0.981477274555806
# The score on the test dataset:
#  0.8832142185785542

##### doing the tuning 
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [8, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 1.0]
}
random_search = RandomizedSearchCV(estimator=forest, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, Y_train)
best_forest = random_search.best_estimator_
print("This is the current model", best_forest)
# 2. Use this trained model to score your test data

# Save the model to a file
joblib.dump(best_forest, 'house_price_model.joblib')
print("Model saved successfully!")