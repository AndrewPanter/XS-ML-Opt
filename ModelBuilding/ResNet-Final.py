# %%
# Build and train the ResNet model, with the hyperparameter values found
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
from tensorflow.keras.metrics import R2Score
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
# Build ResNet Model
def ResidualBlock(input, filters, kernel_size=3, stride=1, padding='same'):
    conv1 = Conv1D(filters, kernel_size, strides=stride, padding=padding, activation='relu')(input)
    conv2 = Conv1D(filters, kernel_size, strides=stride, padding=padding, activation='relu')(conv1)

    x = Conv1D(filters, 1, strides=stride, padding=padding)(input)

    x = Add()([x, conv2])
    x = Activation('relu')(x)

    return x

def build_resnet(layers, nodes, dropout_rate, learning_rate, input_shape=(399, 1)):
    inputs = Input(shape=input_shape)
    x = Conv1D(nodes, 3, padding='same', activation='relu')(inputs)
    for _ in range(layers):
        x = ResidualBlock(x, nodes, kernel_size=3)
    
    x = GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='mse', optimizer=Adam(learning_rate), metrics=['mae', 'R2Score'])
    return model

def train_evaluate(model, batch_size, epochs, X_train, ytrain, X_val, yval):
    history = model.fit(X_train, ytrain, epochs=epochs, validation_data=(X_val, yval), batch_size=batch_size, verbose=0)
    loss, mae, r2 = model.evaluate(X_test, ytest)
    return mae, r2, history

# hyperparameter values found during tuning
layers = 6
nodes = 64
dropout_rate = 0.13360169529817775
batch_size = 128
learning_rate = 0.007790270266029196

# build and train final model
final_model = build_resnet(layers, nodes, dropout_rate, learning_rate)

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', 'R2Score'])
callback = ModelCheckpoint('checkpoints/resnet_best_model.keras', 
                                      monitor='val_mae', 
                                      save_best_only=True)
history = final_model.fit(X_train, ytrain, epochs=500, batch_size=batch_size, validation_split=0.35, callbacks=[callback])
final_loss, final_mae, final_r2 = final_model.evaluate(X_test, ytest)
y_pred = final_model.predict(X_test)

# %%
# Evaluate on test data
resnet_test_loss, resnet_test_mae, resnet_test_r2 = final_model.evaluate(X_test, ytest)
print("Test loss: ", resnet_test_loss, "Test MAE: ", resnet_test_mae, "Test R2:", resnet_test_r2)
y_pred_test = final_model.predict(X_test)
y_pred_train = final_model.predict(X_train)

# make plots
train_err=history.history['mae']
val_err=history.history['val_mae']
plt.figure()
plt.plot(train_err, label='Training')
plt.plot(val_err, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig('MAE_plot_ResNet.png')

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_ResNet.png')

plt.figure()
plt.scatter(y_pred_test, ytest, label = 'Testing Data')
plt.scatter(y_pred_train, ytrain, label = 'Traininging Data')
plt.xlabel('Predicted Outcome')
plt.ylabel('Actual Outcome')
plt.legend()
plt.show()
plt.savefig('diagonal_validation_resnet.png')