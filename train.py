import numpy as np
import pandas as pd
from model import xgb_model
from data_transformation import prep, split

import matplotlib.pyplot as plt
from pathlib import Path
import pickle





def train(df, to_predict = 'Artikel3', plot = False):

    train ,test = split(df)
    model = xgb_model(train,test, to_predict = to_predict, plot = plot)
    filename = Path().resolve()/f"models/model_XGBOOST_{to_predict}.sav"
    pickle.dump(model, open(filename, 'wb'))
    return model





def pipeline(df, to_predict = 'Artikel3', plot = False):
    
    df_feat = prep(df)
    xgmodel = train(df_feat, to_predict = to_predict)

    return xgmodel

if __name__ == "__main__":
    df = pd.read_excel('data/Zeitreihen_2Artikel.xlsx')  
    model = pipeline(df, to_predict = 'Artikel3', plot = False)