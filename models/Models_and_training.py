#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, utils, backend as K
from datetime import datetime
from tensorflow.keras.models import load_model


# In[2]:


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[3]:


train_x = np.loadtxt('new_6k_x_train_2nd_set_only_Grid')
test_x= np.loadtxt('new_6k_x_test_2nd_set_only_Grid')
train_y = np.loadtxt('new_6k_y_train_2nd_set_only_Grid')
test_y = np.loadtxt('new_6k_y_test_2nd_set_only_Grid')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))


# In[4]:


def R2(y, y_hat):
    ss_res =  K.sum(K.square(y - y_hat)) 
    ss_tot = K.sum(K.square(y - K.mean(y))) 
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )


# In[5]:


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(tf.timestamp() - self.timetaken)
        self.epochs.append(epoch)
    def on_train_end(self,logs = {}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(self.epochs, self.times, 'ro')
        for i in range(len(self.epochs)):
          j = self.times[i].numpy()
          if i == 0:
            plt.text(i, j, str(round(j, 3)))
          else:
            j_prev = self.times[i-1].numpy()
            plt.text(i, j, str(round(j-j_prev, 3)))
        plt.savefig(datetime.now().strftime("%Y%m%d%H%M%S") + ".png")


# In[6]:


def train_test(model,num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y):
    training = model.fit(x=train_x, 
                     y=train_y, 
                     batch_size=32, 
                     epochs=num_epochs, 
                     shuffle=True, 
                     verbose=0, 
                     validation_split=0.2,
                     callbacks = [timecallback()])
    
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)] 
    
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.set(title="Training")
    ax11 = ax.twinx()
    ax.plot(training.history['loss'], color='black')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.set(title="Validation")
    ax22 = ax.twinx()
    ax.plot(training.history['val_loss'], color='black')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel("Score", color="steelblue")
    plt.show()
    
    evaluation = model.evaluate(x=test_x, y=test_y, verbose=0)
    print("Evaluation Result:")
    for metric in model.metrics_names:
        print(f"{metric}: {evaluation[model.metrics_names.index(metric)]:.4f}")


# In[7]:


def model_1_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model


# In[8]:


def model_1_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[9]:


def model_2_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model


# In[10]:


def model_2_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[11]:


def model_3_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model

def model_3_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[12]:


def model_4_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model

def model_4_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[13]:


def model_5_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model

def model_5_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[14]:


def model_6_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model

def model_6_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[15]:


def model_7_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model

def model_7_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[16]:


def model_8_mae(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_absolute_error', 
                  metrics=[R2])
    return model

def model_8_mse(input_d):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh',input_shape=[input_d]),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4)
    ])
    
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=[R2])
    return model


# In[17]:


model = model_1_mse(8)
train_test(model,  num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_6k_new.h5')
print('Model Saved!')


# In[18]:


model = model_1_mae(8)
train_test(model,  num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_6k_new.h5')
print('Model Saved!')


# In[19]:


model = model_2_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_2_mse_6k_new.h5')
print('Model Saved!')

model = model_2_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_2_mae_6k_new.h5')
print('Model Saved!')


# In[20]:


model = model_3_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_3_mse_6k_new.h5')
print('Model Saved!')

model = model_3_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_3_mae_6k_new.h5')
print('Model Saved!')


# In[21]:


model = model_4_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_4_mse_6k_new.h5')
print('Model Saved!')

model = model_4_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_4_mae_6k_new.h5')
print('Model Saved!')


# In[22]:


model = model_5_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_5_mse_6k_new.h5')
print('Model Saved!')

model = model_5_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_5_mae_6k_new.h5')
print('Model Saved!')


# In[23]:


model = model_6_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_6_mse_6k_new.h5')
print('Model Saved!')

model = model_6_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_6_mae_6k_new.h5')
print('Model Saved!')


# In[24]:


model = model_7_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_7_mse_6k_new.h5')
print('Model Saved!')

model = model_7_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_7_mae_6k_new.h5')
print('Model Saved!')


# In[25]:


model = model_8_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_8_mse_6k_new.h5')
print('Model Saved!')

model = model_8_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_8_mae_6k_new.h5')
print('Model Saved!')


# for 600k size

# In[26]:


train_x = np.loadtxt('new_600k_x_train_2nd_set_only_Grid')
test_x= np.loadtxt('new_600k_x_test_2nd_set_only_Grid')
train_y = np.loadtxt('new_600k_y_train_2nd_set_only_Grid')
test_y = np.loadtxt('new_600k_y_test_2nd_set_only_Grid')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))


# In[27]:


model = model_1_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new.h5')
print('Model Saved!')


# In[28]:


model = model_2_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_2_mse_600k_new.h5')
print('Model Saved!')

model = model_2_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_2_mae_600k_new.h5')
print('Model Saved!')


# In[29]:


model = model_3_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_3_mse_600k_new.h5')
print('Model Saved!')

model = model_3_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_3_mae_600k_new.h5')
print('Model Saved!')


# In[30]:


model = model_4_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_4_mse_600k_new.h5')
print('Model Saved!')

model = model_4_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_4_mae_600k_new.h5')
print('Model Saved!')


# In[31]:


model = model_5_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_5_mse_600k_new.h5')
print('Model Saved!')

model = model_5_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_5_mae_600k_new.h5')
print('Model Saved!')


# In[32]:


model = model_6_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_6_mse_600k_new.h5')
print('Model Saved!')

model = model_6_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_6_mae_600k_new.h5')
print('Model Saved!')


# In[33]:


model = model_7_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_7_mse_600k_new.h5')
print('Model Saved!')

model = model_7_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_7_mae_600k_new.h5')
print('Model Saved!')


# In[34]:


model = model_8_mse(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_8_mse_600k_new.h5')
print('Model Saved!')

model = model_8_mae(8)
train_test(model, num_epochs = 50, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_8_mae_600k_new.h5')
print('Model Saved!')


# test normalization

# In[35]:


train_x = np.loadtxt('x_train_normalized_min_max.txt')
test_x= np.loadtxt('x_test_normalized_min_max.txt')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))


# In[36]:


model = model_1_mse(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_min_max.h5')
print('Model Saved!')


# In[37]:


model = model_1_mae(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_min_max.h5')
print('Model Saved!')


# In[38]:


train_x = np.loadtxt('x_train_standardized_z_score.txt')
test_x= np.loadtxt('x_test_standardized_z_score.txt')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))

model = model_1_mse(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_z_score.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_z_score.h5')
print('Model Saved!')


# In[39]:


train_x = np.loadtxt('x_train_robust_scaled.txt')
test_x= np.loadtxt('x_test_robust_scaled.txt')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))

model = model_1_mse(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_robust_scaled.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 400, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_robust_scaled.h5')
print('Model Saved!')


# In[40]:


train_x = np.loadtxt('x_train_normalized_unit_vector.txt')
test_x= np.loadtxt('x_test_normalized_unit_vector.txt')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))

model = model_1_mse(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_normalized_unit_vector.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 200, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_normalized_unit_vector.h5')
print('Model Saved!')


# In[41]:


train_x = np.loadtxt('x_train_scaled_max_abs.txt')
test_x= np.loadtxt('x_test_scaled_max_abs.txt')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))

model = model_1_mse(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_scaled_max_abs.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 200, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_scaled_max_abs.h5')
print('Model Saved!')


# In[42]:


train_x = np.loadtxt('new_600k_x_train_2nd_set_only_Grid')
test_x = np.loadtxt('new_600k_x_test_2nd_set_only_Grid')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))

model = model_1_mse(8)
train_test(model, num_epochs = 100, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_noscaled.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 200, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_noscaled.h5')
print('Model Saved!')


# In[43]:


train_x = np.loadtxt('new_600k_x_train_2nd_set_only_Grid')
test_x = np.loadtxt('new_600k_x_test_2nd_set_only_Grid')
train_y = np.loadtxt('new_600k_y_train_multiplied.txt')
test_y = np.loadtxt('new_600k_y_test_multiplied.txt')
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(np.max(train_x, axis=0))
print(np.max(train_y, axis=0))
print(np.max(test_x, axis=0))
print(np.max(test_y, axis=0))

model = model_1_mse(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mse_600k_new_noscaled_5x.h5')
print('Model Saved!')

model = model_1_mae(8)
train_test(model, num_epochs = 500, train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y)
model.save('Model_1_mae_600k_new_noscaled_5x.h5')
print('Model Saved!')


# In[44]:


print(np.mean(train_y, axis = 0))
print(np.mean(train_x, axis = 0))


# In[45]:


model = load_model('Model_1_mse_600k_new_noscaled.h5', custom_objects={'R2': R2})

data = np.array([[0.01,0.01,50*np.pi/180, 4.52,3.03, 0.4,0.29,0.2]])
data = np.array([ np.mean(train_x, axis = 0)])
res = model.predict(data)
print(res)


# In[ ]:




