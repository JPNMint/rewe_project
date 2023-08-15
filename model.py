import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import cross_val_predict,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def xgb_model(train, test, to_predict = 'Artikel3', plot = False):
    train = train.drop('DATUM', axis = 1)
    test = test.drop('DATUM', axis = 1)
    X_train = train.loc[:, ~train.columns.isin(['Artikel3', 'Artikel5', 'DATUM'])]



    #X_train = train.loc[:, train.columns != to_predict]
    y_train = train[to_predict]
    X_test = test.loc[:, ~test.columns.isin(['Artikel3', 'Artikel5', 'DATUM'])]

    #X_test = test.loc[:, test.columns != to_predict]
    y_test = test[to_predict]
    #scaler = StandardScaler()
    #X_trainScaled = scaler.fit_transform(X_train)

    model = xgb.XGBRegressor(random_state=42)
    

    # Define a grid of hyperparameters to search through
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 800],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5], # Maximum depth of a tree 
        'subsample': [0.8, 0.9, 1.0], #Subsample ratio of the training instances. if 0.75, 25% of the data is unused by that learner 
        'colsample_bytree': [0.8, 0.9, 1.0] 
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)


    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator from the grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Best Mean Squared Error: {mse}")
    print("Best Parameters:", best_params)


    if plot:

        # Plot true vs. predicted values
        plt.scatter(y_test, y_pred)
        plt.xlabel("True Sales")
        plt.ylabel("Predicted Sales")
        plt.title("True vs. Predicted Sales")
        plt.show()

    return best_model