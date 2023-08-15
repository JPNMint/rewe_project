import numpy as np
import pandas as pd
from model import xgb_model, outlier_detector
from data_transformation import prep, split

import matplotlib.pyplot as plt
from pathlib import Path
import pickle





def train(df, to_predict = 'Artikel3', plot = False, winsorization = True, type = 'Regressor'):

    train ,test = split(df)

    if type == 'Regressor':
        model = xgb_model(train,test, to_predict = to_predict, plot = plot, winsorization = True)
        if winsorization:
            filename = Path().resolve()/f"models/model_XGBOOST_{to_predict}_winsorize.sav"
        else:
            filename = Path().resolve()/f"models/model_XGBOOST_{to_predict}.sav"
        pickle.dump(model, open(filename, 'wb'))
    if type == 'Outlier':
        model = outlier_detector(train, test, to_predict = to_predict, plot = False)
        filename = Path().resolve()/f"models/model_outlier_detector_{to_predict}.sav"
        pickle.dump(model, open(filename, 'wb'))

    
    return model





def pipeline(df, to_predict = 'Artikel3', plot = False, winsorization = True, type = 'Regressor'):
    if type == 'Regressor':
        df_feat = prep(df)
        model = train(df_feat, to_predict = to_predict, type = type)
    if type == 'Outlier':
        df_feat = prep(df)
        model = train(df_feat, to_predict = to_predict, type = type)


    return model

if __name__ == "__main__":
    df = pd.read_excel('data/Zeitreihen_2Artikel.xlsx')  
    #model = pipeline(df, to_predict = 'Artikel5', plot = False,  winsorization = True, train = 'Regressor)

    model = pipeline(df, to_predict = 'Artikel5', plot = False, type = 'Outlier')
    #model = pipeline(df, to_predict = 'Artikel3', plot = False, type = 'Outlier')