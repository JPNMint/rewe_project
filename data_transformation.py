import numpy as np
import pandas as pd




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
