import time
start = time.perf_counter()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras.models import load_model

df = pd.read_csv('fake_reg.csv')
print(df.info)

plt.figure(1)
sns.pairplot(df)
plt.show()

# TensorFlow cannot use panda df do use '.values' to change to np array
X = df[['feature1','feature2']].values
y = df['price'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=42)

#Normalise and scale data

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model training

#model1_method = Sequential([Dense(4,activation='relu'),Dense(2,activation='relu'),Dense(1)])
model2_method = Sequential()

model2_method.add(Dense(4,activation='relu'))
model2_method.add(Dense(4,activation='relu'))
model2_method.add(Dense(4,activation='relu'))

model2_method.add(Dense(1))

model2_method.compile(optimizer='rmsprop',loss='mse')

model2_method.fit(x=X_train,y=y_train,epochs=350)

loss_df = pd.DataFrame(model2_method.history.history)
print(loss_df)

plt.figure(2)
loss_df.plot()
plt.show()

# Model validation

model2_method.evaluate(X_test,y_test)
test_predictions  = model2_method.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))

pred_df = pd.DataFrame(y_test,columns=['Test True Y'])
pred_df = pd.concat([pred_df,test_predictions],axis=1)
pred_df.columns = ['Test True Y','Model Predictions']
print(pred_df)

plt.figure(3)
sns.scatterplot(x='Test True Y',y='Model Predictions',data=pred_df)
plt.show()

print('MAE:',+mean_absolute_error(pred_df['Test True Y'],pred_df['Model Predictions']))
print('MSE:',+mean_squared_error(pred_df['Test True Y'],pred_df['Model Predictions']))
print('RMSE:',+mean_squared_error(pred_df['Test True Y'],pred_df['Model Predictions'])**0.5)

# Saving model
model2_method.save('my_first_model.h5')
later_model = load_model('my_first_model.h5')

# Making random Prediction
input_values = [[998,1000]]

input_pred = later_model.predict(scaler.transform(input_values))
print('$'+str(round(float(input_pred),2)))

end = time.perf_counter()

print('runtime: '+str(end-start)+' s')
