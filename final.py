# Importing necessary packages for the project
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Reading in the data from the csv files provided
train = pd.read_csv('final_train-6.csv')
test_features = pd.read_csv('final_test-3.csv')

# Counting the number of missing values
train.isna().sum()

# Some additional exploratory data analysis to see the datatypes of columns
train.info()

# Splitting training data into labels and features
train_label = train['price']
train_features = train.drop('price', axis=1)

# Creating numerical dummies for the categorical variables
categorical_features = train_features[['cut', 'color', 'clarity']]
categorical_dummies = pd.get_dummies(categorical_features, drop_first=True)

# Adding these numerical dummies into the og dataframe
train_features = train_features.drop(['cut', 'color', 'clarity'], axis=1)
train_features = pd.concat([train_features, categorical_dummies], axis=1)

"""
I chose to use K nearnerst neighbor imputer to create the missing values
in the data
Here we create these new values and concat with our original df
"""
imputer = KNNImputer(n_neighbors=3)
x = imputer.fit_transform(train_features)
train_features = pd.DataFrame(x, columns=train_features.columns,
                              index=train_features.index)

# Check to see that all of the features have no missing values
train.info()

# This is a small helper function to see the results of our regression
def check_results(observation, prediction):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(observation, prediction)
    ax.plot([0, 18000], [0, 18000], color='red')
    ax.set_ylim(0, prediction.max())
    ax.set_ylabel('observed value')
    ax.set_ylabel('predicted value')

# Creating and training a random forest regressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=380)
forest_reg.fit(train_features, train_label)

# Create and test predictions from the training set
forest_prediction = forest_reg.predict(train_features)

# Check the results and the mean squared error to ensure model works
check_results(train_label, forest_prediction)
mse = mean_squared_error(train_label, forest_prediction)
print(np.sqrt(mse))

# Now we move onto predicting the price of the given test set
# Here we once again create numerical dummies for the categorical variables
cat_features = test_features[['cut', 'color', 'clarity']]
cat_dummies = pd.get_dummies(cat_features, drop_first=True)

# Now we concat these dummies onto the test features df
test_features = test_features.drop(['cut', 'color', 'clarity'], axis=1)
test_features = pd.concat([test_features, cat_dummies], axis=1)

# Once again we use the KNN imputer to add the missing values
y = imputer.fit_transform(test_features)
test_features = pd.DataFrame(y, columns=test_features.columns,
                             index=test_features.index)

# Here I predict the price of the test features
test_prediction_forest = pd.Series(forest_reg.predict(test_features))

# Lastly, we create the payload in the desired format: product id, price
payload = pd.concat([test_features['product_id'], test_prediction_forest],
                    axis=1)
payload = payload.rename({0: 'price'}, axis=1)
pd.to_numeric(payload['product_id'], downcast='integer')

# This creates a csv file of the payload in the current working directory
payload.to_csv(os.path.join(os.getcwd(), "prediction.csv"))
