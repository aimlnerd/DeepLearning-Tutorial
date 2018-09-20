import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, MaxPooling1D, GlobalMaxPool1D

D = np.random.rand(10, 6, 10)

model = Sequential()
model.add(LSTM(16, input_shape=(6, 10), return_sequences=True))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(LSTM(10))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='sgd')

# print the summary to see how the dimension change after the layers are
# applied

print(model.summary())

# try a model with MaxGlobalPooling1D now

model = Sequential()
# Input shape (batch_size=None, steps=6, features=16) None represents batch_size
model.add(LSTM(16, input_shape=(6, 10), return_sequences=True))
# LSTM output shape (batch_size=None, steps=6, hidden_size=16)
model.add(GlobalMaxPool1D())
# GlobalMaxPool1D output shape (batch_size, hidden_size=16) # i.e. take max of hidden_size across the steps
# i.e. horizontally across the steps_size number of LSTM blocks
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='sgd')

print(model.summary())