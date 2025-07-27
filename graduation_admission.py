import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
df = pd.read_csv(r'C:\Users\ssp1_\OneDrive\Desktop\majorprojects\Graduate-Admission-Prediction-using-ANN-main\Admission_Predict.csv')
df.head()
df.shape
df.info()
df.duplicated().sum()
df.drop(columns=['Serial No.'],inplace=True)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Sequential()

model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()
def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - (ss_res / ss_tot)
model.compile(loss='mean_squared_error',optimizer='Adam',metrics=[r2_score])
history = model.fit(X_train_scaled,y_train,epochs=10,validation_split=0.2)
y_pred = model.predict(X_test_scaled)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['r2_score'])
plt.plot(history.history['val_r2_score'])
