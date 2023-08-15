import numpy as np
import pandas as pd
from model import xgb_model
from data_transformation import prep

import matplotlib.pyplot as plt



df = pd.read_excel('data/Zeitreihen_2Artikel.xlsx')  
df_feat = prep(df)

print(df_feat.dtypes)
test = df_feat[df_feat['year'] == 2017]
train = df_feat[df_feat['year'] != 2017]
#print( np.isnan(train).values.sum() )

model = xgb_model(train,test)
#cols = test.columns
#cols = cols.remove(['Artikel3', 'DATUM'])
test_new = test.loc[:, ~test.columns.isin(['Artikel3', 'Artikel5', 'DATUM'])]
y_pred = model.predict(test_new)

test['predicted'] = y_pred  


plt.figure(figsize=(12, 6))
#plt.plot(train['DATUM'], train['Artikel3'], label='Actual Sales')

plt.plot(test['DATUM'], test['Artikel3'], label='Actual Sales')
plt.plot(test['DATUM'] , test['predicted'], label='Predicted Sales', linestyle='dashed')
plt.axvline(x=pd.to_datetime('2017-01-01'), color='r', linestyle='--', label='Prediction Start')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales Timeline')
plt.legend()
plt.show()