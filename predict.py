import numpy as np
import pandas as pd
from model import xgb_model
from data_transformation import prep
from train import train, split
import matplotlib.pyplot as plt
import pickle 

def predict(test, type = 'Regressor', filename = 'models/model_XGBOOST_Artikel3.sav', outlier_predictor = 'Artikel3'):

    loaded_model = pickle.load(open(filename, 'rb'))
    if type == 'Regressor':
        #test set 2017 to predict 
        test_new = test.loc[:, ~test.columns.isin(['Artikel3', 'Artikel5', 'DATUM'])]
        y_pred = loaded_model.predict(test_new)
    if type == 'Outlier':
        x_remover_list = ['is_outlier', 'DATUM']
        if outlier_predictor == 'Artikel3':
            x_remover_list.append('Artikel5')
        if outlier_predictor == 'Artikel5':
            x_remover_list.append('Artikel3')
        test_new = test.loc[:, ~test.columns.isin(x_remover_list)]
        y_pred = loaded_model.predict(test_new)

    return y_pred



def plot_timeline(test, predicted_value = 'Artikel3', variant = '2017', outlier = False):
    #Visualize 
    plt.figure(figsize=(12, 6))
    if outlier:
        outlier_date = test[test['outlier'] == 1]['DATUM']
    
    #plt.plot(train['DATUM'], train['Artikel3'], label='Actual Sales')
    if variant == '2017':
        plt.plot(test['DATUM'], test[predicted_value], label='Actual Sales')
        plt.plot(test['DATUM'] , test['predicted'], label='Predicted Sales', linestyle='dashed')
        if outlier:
            for i in outlier_date:
                plt.axvline(x=i, color='grey', linestyle=':', label='')




        plt.axvline(x=pd.to_datetime('2017-01-01'), color='r', linestyle='--', label='Prediction Start')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Actual vs. Predicted Sales Timeline - {predicted_value}')
        plt.legend()
        plt.show()

    if variant == 'all':
        plt.plot(train['DATUM'], train[predicted_value], label='Actual Sales')
        plt.plot(test['DATUM'], test[predicted_value], label='Actual Sales')
        plt.plot(test['DATUM'] , test['predicted'], label='Predicted Sales', linestyle='dashed')
        plt.axvline(x=pd.to_datetime('2017-01-01'), color='r', linestyle='--', label='Prediction Start')
        if outlier:
            for i in outlier_date:
                plt.axvline(x=i, color='grey', linestyle=':', label='')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Actual vs. Predicted Sales Timeline - {predicted_value}')
        plt.legend()
        plt.show()
#plot_timeline(test, variant = '2017')


if __name__ == "__main__":

    df = pd.read_excel('data/Zeitreihen_2Artikel.xlsx')  

    df_feat = prep(df)
    train, test = split(df_feat)
    y_pred = predict(test, filename = 'models/model_XGBOOST_Artikel3_winsorize.sav')
    outlier = predict(test, filename = 'models/model_outlier_detector_Artikel3.sav', type = 'Outlier', outlier_predictor='Artikel3')
    
    test['predicted'] = y_pred 
    test['outlier'] = outlier 
    #print(test)
    plot_timeline(test, variant = '2017', outlier = True)
