# %%
# Build and train the CNN model, with the hyperparameter values found
# through tuning hardcoded into building the model

# %%
# Import packages and load data
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

#Keras/TensorFlow
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Add, ReLU, BatchNormalization, Activation, GlobalAveragePooling1D, Dropout
from tensorflow.keras.metrics import R2Score, MeanAbsolutePercentageError
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from skopt import gp_minimize

# check if the GPU or CPU is being used 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")
    
# load and shape data
data = np.loadtxt('compiled_data.txt', delimiter=',') 

xdata = data[:, 0:-1]
ydata = data[:, -1]
xtrain, xtest, ytrain, ytest=train_test_split(xdata, ydata, test_size=0.2, random_state=42)

X_train = xtrain.reshape(-1, 399, 1)
X_test = xtest.reshape(-1, 399, 1)

# %%
# Build CNN Model
def build_CNN(layers, nodes, dropout_rate, learning_rate, input_shape=(399, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(nodes, kernel_size=3, activation='relu'))
    for _ in range(layers - 1):
        model.add(Conv1D(nodes, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2)) #2
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(nodes, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(learning_rate), metrics=['mae', 'R2Score'])
    return model

def train_evaluate(model, batch_size, epochs, X_train, ytrain, X_val, yval):
    history = model.fit(X_train, ytrain, epochs=epochs, validation_data=(X_val, yval), batch_size=batch_size, verbose=0)
    loss, mae, r2 = model.evaluate(X_test, ytest)
    return mae, r2, history

# hyperparameter values found with tuning
layers = 1
nodes = 128
dropout_rate = 0.426514149825001
batch_size = 128
learning_rate = 0.0007413451805059539

# build and train model
final_model = build_CNN(layers, nodes, dropout_rate, learning_rate)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', 'R2Score'])
callback = ModelCheckpoint('checkpoints/cnn_best_model.keras', 
                                      monitor='val_mae', 
                                      save_best_only=True)
final_history = final_model.fit(X_train, ytrain, epochs=500, batch_size=batch_size, validation_split=0.35, callbacks=[callback])
final_loss, final_mae, final_r2 = final_model.evaluate(X_test, ytest)
y_pred_test = final_model.predict(X_test)
y_pred_train = final_model.predict(X_train)

# print results
print("Test loss: ", final_loss, "Test MAE: ", final_mae, "Test R2:", final_r2)

# make plots
train_err=final_history.history['mae']
val_err=final_history.history['val_mae']
plt.figure()
plt.plot(train_err, label='Training')
plt.plot(val_err, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig('MAE_plot_CNN-Final.png')

train_loss = final_history.history['loss']
val_loss = final_history.history['val_loss']
plt.figure()
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_CNN-Final.png')

plt.figure()
plt.scatter(y_pred_test, ytest, label = 'Testing Data')
plt.scatter(y_pred_train, ytrain, label = 'Traininging Data')
plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')
plt.legend()
plt.show()
plt.savefig('diagonal_validation_CNN-Final.png')
