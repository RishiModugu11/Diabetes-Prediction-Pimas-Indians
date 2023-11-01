import numpy as np
import pandas as pd
from numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("pima-indians-diabetes.csv")
df_predict = df.iloc[600:,:]
df_train = df.iloc[:600,:]

train_dataset = df_train.to_numpy()
predict_dataset = df_predict.to_numpy()

# split into input (X) and output (Y) variables using the train and predict datasets
X_train = train_dataset[:,0:8]
Y_train = train_dataset[:,8]

X_predict = predict_dataset[:,0:8]
Y_predict = predict_dataset[:,8]

# define the keras model as sequential using relu and sigmoid activation functions
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model using binary crossentropy and adam gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the data separated for training the model
model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)

# evaluate the keras model with the training data and new data
_, accuracy_train = model.evaluate(X_train, Y_train)
print('Train Accuracy: %.2f' % (accuracy_train*100))
_, accuracy = model.evaluate(X_predict, Y_predict)
print('Prediction Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model based on new data
predictions = (model.predict(X_predict) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(15):
    print('%s => %d (expected %d)' % (X_predict[i].tolist(), predictions[i], Y_predict[i]))