import numpy as np
import pandas as pd

import holidays
pd.options.mode.chained_assignment = None  # default='warn'

def aggregate(df, resample = 'week'):

    df['DATUM'] = pd.to_datetime(df['DATUM'])  # Convert the 'date' column to datetime
    df = df.drop(['Day','Holidays'], axis = 1)
    df.set_index('DATUM', inplace=True)
    if resample not in ['week', 'month']:
        raise ValueError("Choose week or month!")

    if resample == 'week':
        print('Grouping mean value per week')

        #resample by week
        df = df.resample('W-MON').mean()

        #Reset index to get week number and year, prob can be done per index instead aswell
        df = df.reset_index()
        df['Week_number'] = df['DATUM'].dt.isocalendar().week
        df['Year'] = df['DATUM'].dt.isocalendar().year

    if resample == 'month':
        print('Grouping mean value per month')

        #resample by week
        df = df.resample('M').mean()

        #Reset index to get week number and year, prob can be done per index instead aswell
        df = df.reset_index()

        df['Week_number'] = df['DATUM'].dt.isocalendar().week
        df['Year'] = df['DATUM'].dt.isocalendar().year

    return df 

###
def split(df, Years = [2017]):

    #train test split, test is year list
    # Years in Years will be excluded in training
    test = df[df['year'].isin(Years)]
    train = df[~df['year'].isin(Years)]
    
    return train ,test

def prep(df):

    df = df.dropna()
    ## add interpolation for the 13 missing values ?
    #df = df.interpolate(method = 'time')
    df['DATUM']  =  pd.to_datetime(df['DATUM']) 

    #add day 
    df['dayofweek'] = df['DATUM'].dt.dayofweek

    #drop sunday 

    df = df[df['dayofweek'] != 6]

    #flag holidays
    at_holidays = holidays.Austria(years=df['DATUM'].dt.year.unique())
    df['Holidays'] = df['DATUM'].dt.date.isin(at_holidays)
    #drop holidays
    df = df[df['Holidays'] != 1]    


    
    # get various date features
    df['Week_number'] = df['DATUM'].dt.isocalendar().week.astype('int32')
    df['dayofweek'] = df['DATUM'].dt.dayofweek
    df['quarter'] = df['DATUM'].dt.quarter
    df['month'] = df['DATUM'].dt.month
    df['year'] = df['DATUM'].dt.year
    df['dayofyear'] = df['DATUM'].dt.dayofyear
    df['dayofmonth'] = df['DATUM'].dt.day


    return df





#df_new = pd.read_excel('data/Zeitreihen_2Artikel.xlsx')  
