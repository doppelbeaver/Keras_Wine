import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import cohen_kappa_score


white = pd.read_csv('winequality-white.csv',sep=";")
red = pd.read_csv('winequality-red.csv',sep=";")

print(white)
print(red)

red['type'] = 1
white['type'] = 0

wines = red.append(white,ignore_index=True)
print(wines)
# Specify the data
X=wines.ix[:,0:11]
# Specify the target labels and flatten the array
y=np.ravel(wines.type)
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)
print(scaler)
X_train = scaler.transform(X_train)
print(X_train)
X_test = scaler.transform(X_test)
print(X_test)

model = Sequential()

model.add(Dense(12, activation="relu", input_shape=(11,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)
y_pred = model.predict(X_test)


score = model.evaluate(X_test, y_test,verbose=1)
print(score)