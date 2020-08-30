import math
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

data = pd.read_csv('BitcoinHistoricalData-Investing.comIndia.csv')
df = data[['Open Price','High Price','Low Price', 'Price']]

df['open'] = df['Open Price']
df['high'] = df['High Price']
df['low'] = df['Low Price']
df['price'] = df['Price']


df['label'] = df['price'].shift(-3)

X = np.array(df[['open','high','low']])
X_lately = X[-7:]
X = X[:-7]
y = np.array(df['label'])
y = y[:-7]

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)

print("X_train, X_test, y_train, y_test",X_train.shape, x_test.shape, Y_train.shape, y_test.shape)
clf = LinearRegression()
clf.fit(X_train, Y_train)
confidence = clf.score(x_test, y_test)
print('confidence:', confidence)
forecast_set = clf.predict(X_lately)
print("forecast_set",forecast_set)

