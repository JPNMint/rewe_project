import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import cross_val_predict,GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import pmdarima as pm
from math import sqrt

def xgb_model(train, test, to_predict = 'Artikel3', plot = False, winsorization = True):
    train = train.drop('DATUM', axis = 1)
    test = test.drop('DATUM', axis = 1)
    X_train = train.loc[:, ~train.columns.isin(['Artikel3', 'Artikel5', 'DATUM'])]



    #get train test set
    y_train = train[to_predict]
    X_test = test.loc[:, ~test.columns.isin(['Artikel3', 'Artikel5', 'DATUM'])]
    y_test = test[to_predict]

    #scaling?
    #scaler = StandardScaler()
    #X_trainScaled = scaler.fit_transform(X_train)


    #https://www.statology.org/winsorize/
    if winsorization:
        y_train = winsorize(y_train, limits = [0.02, 0.05])

    model = xgb.XGBRegressor(random_state=4122)
    
    # Define grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300, 500, 700, 900, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth':  [3, 4, 6, 5, 10], # Maximum depth of a tree 
        'subsample': [0.8, 0.9, 1.0], #Subsample ratio of the training instances. if 0.75, 25% of the data is unused by that learner 
        'colsample_bytree':  [0.3, 0.5, 0.8, 0.9, 1.0] 
    }
    #CV splits with fixed time intervals
    cv = TimeSeriesSplit(n_splits=4, test_size=100)

    #GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

    #fitting training data
    grid_search.fit(X_train, y_train)

    # Get best params & model from the grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    #score
    mse = mean_squared_error(y_test, y_pred)
    print(f"Best Root Mean Squared Error: {sqrt(mse)}")
    print("Best Parameters:", best_params)


    if plot:

        # Plot true vs. predicted values
        plt.scatter(y_test, y_pred)
        plt.xlabel("True Sales")
        plt.ylabel("Predicted Sales")
        plt.title("True vs. Predicted Sales")
        plt.show()

    return best_model




def get_label(df, outlier_threshold = 1, to_predict = "Artikel3"):
    #calc z score
    #https://www.statisticshowto.com/probability-and-statistics/z-score/
    z_scores = np.abs((df[to_predict] - df[to_predict].mean()) / df[to_predict].std())
    df['is_outlier'] = z_scores > outlier_threshold
    return df

def outlier_detector(train, test, to_predict = 'Artikel3', outlier_threshold = 1, plot = False):

    train = train.drop('DATUM', axis = 1)
    test = test.drop('DATUM', axis = 1)

    # Calculate z scores
    train = get_label(train, outlier_threshold = outlier_threshold, to_predict = to_predict )
    test = get_label(test, outlier_threshold = outlier_threshold, to_predict = to_predict )

    x_remover_list = ['is_outlier', 'DATUM', 'Artikel5','Artikel3']


    X_train = train.loc[:, ~train.columns.isin(x_remover_list)]
    y_train = train['is_outlier']

    X_test = test.loc[:, ~test.columns.isin(x_remover_list)]
    y_test = test['is_outlier']

    model = xgb.XGBClassifier(random_state = 2332, objective = 'binary:logistic', eval_metric = 'logloss')
    

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

    y_pred = best_model.predict_proba(X_test)[:, 1]  # Probability of positive class

    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC-AUC Score: {roc_auc}")


    if plot:

        # Plot true vs. predicted values
        plt.scatter(y_test, y_pred)
        plt.xlabel("True Sales")
        plt.ylabel("Predicted Sales")
        plt.title("True vs. Predicted Sales")
        plt.show()

    return best_model





def arima_model(train, to_predict = 'Artikel3'):
    train = train.drop('DATUM', axis = 1)
    
    X_train = train.loc[:, ~train.columns.isin(['Artikel3', 'Artikel5'])]

    y_train = train[to_predict]

    model = pm.auto_arima(y_train, seasonal=True, m=52)
    #preds = model.predict(X_test)

    return model
if __name__ == "__main__":
    df = pd.read_excel('data/Zeitreihen_2Artikel.xlsx')  
