#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, utils, backend as K
from tensorflow.keras.models import load_model

import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# In[2]:


def tanh(x):
    return np.tanh(x)

# Function to perform predictions using the extracted parameters
def custom_predict(input_data, model):
    # Extract model parameters (weights and biases) for each layer
    model_parameters = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        model_parameters.append(layer_weights)
        
    for i, layer_params in enumerate(model_parameters):
        weights, biases = layer_params
        # Compute the linear transformation (z) for this layer
        z = np.dot(input_data, weights) + biases
        # Apply the 'tanh' activation function for all layers except the last one
        if i < len(model_parameters) - 1:
            output_data = tanh(z)
        else:
            # For the last layer, do not apply the 'tanh' activation function
            output_data = z
        # Set the output of this layer as the input for the next layer
        input_data = output_data

    # The final output is the prediction
    return output_data


# In[3]:


# y = y_true y_hat = y_pred
def R2(y, y_hat):
    y = tf.convert_to_tensor(value=y, dtype='float32')
    y_hat = tf.convert_to_tensor(value=y_hat, dtype='float32')
    
    ss_res = K.sum(K.square(y - y_hat)) 
    ss_tot = K.sum(K.square(y - K.mean(y))) 
    r_squared = 1 - ss_res / (ss_tot + K.epsilon())
    
    return r_squared.numpy().astype(np.float32)  # Convert tensor to a float32 value

def RMSE(y, y_hat):
    return np.sqrt(np.mean((y_hat - y)**2))

def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))


# In[4]:


x1 = np.loadtxt('updated_new_input1.txt')
x2 = np.loadtxt('updated_new_input2.txt')
y1 = np.loadtxt('updated_new_output1.txt')
y2 = np.loadtxt('updated_new_output2.txt')


# In[5]:


print(x1.shape, y1.shape, x2.shape, y2.shape)


# In[6]:


def test_model(model,x,y):
    evaluation = model.evaluate(x=x, y=y, verbose=0)
    print("Evaluation Result:")
    for metric in model.metrics_names:
        print(f"{metric}: {evaluation[model.metrics_names.index(metric)]:.4f}")


# In[7]:


def test_model_kn_scaled(model,x,y):
    kn = 2.0e8 
    kt = 0.5 * kn
    factor = np.array([kn,kn,kt,1])
    
#     y_pred = model.predict(x, verbose=0) 
    y_pred = custom_predict(x, model)
    y_pred = y_pred * factor
    y_true = y * factor
    r2_f = R2(y_true[:,:3],y_pred[:,:3])
    r2_a = R2(y_true[:,-1],y_pred[:,-1])    
    print('force max:',      np.max(y_pred[:,:3], axis = 0),      'angle max: ', np.max(y_pred[:,-1]))
    print('true force max:', np.max(y_true[:,:3], axis = 0), 'true angle max: ', np.max(y_true[:,-1]))
    
    print('R2 force: ', r2_f, 'R2 angle: ', r2_a)
    print('RMSE force: ', RMSE(y_true[:,:3],y_pred[:,:3]), 'RMSE angle: ', RMSE(y_true[:,-1],y_pred[:,-1]))    
    print('MAE force: ', MAE(y_true[:,:3],y_pred[:,:3]), 'MAE angle: ', MAE(y_true[:,-1],y_pred[:,-1]))    


# In[9]:


def test_model_1e4_scaled(model,x,y,scaler):
    
    pred_factor = np.array([1e4,1e4,1e4,1e-4])
    x = scaler.transform(x)
    
#     y_pred = model.predict(x, verbose=0) 
    y_pred = custom_predict(x, model)
    y_pred = y_pred * pred_factor
    
    kn = 2.0e8 
    kt = 0.5 * kn
    true_factor = np.array([kn,kn,kt,1])
    
    y_true = y * true_factor
    r2_f = R2(y_true[:,:3],y_pred[:,:3])
    r2_a = R2(y_true[:,-1],y_pred[:,-1])    
    
    print('force max:', np.max(y_pred[:,:3], axis = 0), 'angle max: ', np.max(y_pred[:,-1]))
    print('true force max:', np.max(y_true[:,:3], axis = 0), 'true angle max: ', np.max(y_true[:,-1]))
    
    print('R2 force: ', r2_f, 'R2 angle: ', r2_a)
    print('RMSE force: ', RMSE(y_true[:,:3],y_pred[:,:3]), 'RMSE angle: ', RMSE(y_true[:,-1],y_pred[:,-1]))    
    print('MAE force: ', MAE(y_true[:,:3],y_pred[:,:3]), 'MAE angle: ', MAE(y_true[:,-1],y_pred[:,-1]))  


# In[9]:


array_10x3 = np.arange(1, 31).reshape(10, 3)
array_10x3_floats = np.arange(1.5, 31.5, 1).reshape(10, 3)

y_true = array_10x3
y_pred = array_10x3_floats

r2_f = R2(y_true[:,:3],y_pred[:,:3])
r2_a = R2(y_true[:,:-1],y_pred[:,:-1])    
print('R2 force: ', r2_f, 'R2 angle: ', r2_a)
print('RMSE force: ', RMSE(y_true[:,:3],y_pred[:,:3]), 'RMSE angle: ', RMSE(y_true[:,:-1],y_pred[:,:-1]))    
print('MAE force: ', MAE(y_true[:,:3],y_pred[:,:3]), 'MAE angle: ', MAE(y_true[:,:-1],y_pred[:,:-1]))


# In[10]:


for i in range(1,9):
    
    print('test on model ', i ,'\n')
    
    model_name = 'Model_' + str(i) + '_mae_6k_new.h5'
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1')
    test_model_kn_scaled(model,x1,y1)
    print('test on dataset2')
    test_model_kn_scaled(model,x2,y2)
    print('\n')
    
    model_name = 'Model_' + str(i) + '_mse_6k_new.h5'
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1')
    test_model_kn_scaled(model,x1,y1)
    print('test on dataset2')
    test_model_kn_scaled(model,x2,y2)
    print('\n')
    
    model_name = 'Model_' + str(i) + '_mae_600k_new.h5'
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1')
    test_model_kn_scaled(model,x1,y1)
    print('test on dataset2')
    test_model_kn_scaled(model,x2,y2)
    print('\n')
    
    model_name = 'Model_' + str(i) + '_mse_600k_new.h5'
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1')
    test_model_kn_scaled(model,x1,y1)
    print('test on dataset2')
    test_model_kn_scaled(model,x2,y2)
    
    print('\n')


# normlization

# In[11]:


with open('x_scaler_min_max.pkl', 'rb') as x_scaler_min_max_file:
    x_scaler_min_max = pickle.load(x_scaler_min_max_file)

with open('x_scaler_z_score.pkl', 'rb') as x_scaler_z_score_file:
    x_scaler_z_score = pickle.load(x_scaler_z_score_file)

with open('x_scaler_robust.pkl', 'rb') as x_scaler_robust_file:
    x_scaler_robust = pickle.load(x_scaler_robust_file)


# In[12]:


model_list = ['Model_1_mse_600k_new_min_max.h5','Model_1_mae_600k_new_min_max.h5',
              'Model_1_mse_600k_new_z_score.h5','Model_1_mae_600k_new_z_score.h5',
              'Model_1_mse_600k_new_robust_scaled.h5','Model_1_mae_600k_new_robust_scaled.h5']

scaler_list = [x_scaler_min_max,x_scaler_z_score, x_scaler_robust]


# In[13]:


for i in range(3):
    model_name = model_list[2*i]
    scaler = scaler_list[i]
    
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1 for mse')
    test_model_1e4_scaled(model,x1,y1,scaler)
    print('test on dataset2 for mse')
    test_model_1e4_scaled(model,x2,y2,scaler)
    print('\n')
    
    model_name = model_list[2*i + 1]
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1 for mae')
    test_model_1e4_scaled(model,x1,y1,scaler)
    print('test on dataset2 for mae')
    test_model_1e4_scaled(model,x2,y2,scaler)
    print('\n')


# In[9]:


def unit_vector_scaling(data):
    norms = np.linalg.norm(data, axis=1)
    scaled_data = data / norms[:, np.newaxis]
    return scaled_data

def max_absolute_scaling(data):
    max_abs = np.max(np.abs(data), axis=0)
    scaled_data = data / max_abs
    return scaled_data

def no_scale(data):
    return data


# In[10]:


def test_model_1e4_f(model,x,y,f):
    
    pred_factor = np.array([1e4,1e4,1e4,1e-4])
    x = f(x)
    
#     y_pred = model.predict(x, verbose=0) 
    y_pred = custom_predict(x, model)
    y_pred = y_pred * pred_factor
    
    kn = 2.0e8 
    kt = 0.5 * kn
    true_factor = np.array([kn,kn,kt,1])
    
    y_true = y * true_factor
    r2_f = R2(y_true[:,:3],y_pred[:,:3])
    r2_a = R2(y_true[:,-1],y_pred[:,-1])    
    
    print('force max:', np.max(y_pred[:,:3], axis = 0), 'angle max: ', np.max(y_pred[:,-1]))
    print('true force max:', np.max(y_true[:,:3], axis = 0), 'true angle max: ', np.max(y_true[:,-1]))
    
    print('R2 force: ', r2_f, 'R2 angle: ', r2_a)
    print('RMSE force: ', RMSE(y_true[:,:3],y_pred[:,:3]), 'RMSE angle: ', RMSE(y_true[:,-1],y_pred[:,-1]))    
    print('MAE force: ', MAE(y_true[:,:3],y_pred[:,:3]), 'MAE angle: ', MAE(y_true[:,-1],y_pred[:,-1]))


# In[11]:


model_list = ['Model_1_mse_600k_new_normalized_unit_vector.h5','Model_1_mae_600k_new_normalized_unit_vector.h5',
              'Model_1_mse_600k_new_scaled_max_abs.h5','Model_1_mae_600k_new_scaled_max_abs.h5',
              'Model_1_mse_600k_new_noscaled.h5','Model_1_mae_600k_new_noscaled.h5']

f_list = [unit_vector_scaling,max_absolute_scaling,no_scale]


# In[12]:


for i in range(3):
    model_name = model_list[2*i]
    f = f_list[i]
    
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1 for mse')
    test_model_1e4_f(model,x1,y1,f)
    print('test on dataset2 for mse')
    test_model_1e4_f(model,x2,y2,f)
    print('\n')
    
    model_name = model_list[2*i + 1]
    model=load_model(model_name, custom_objects={'R2': R2})
    print('test on dataset1 for mae')
    test_model_1e4_f(model,x1,y1,f)
    print('test on dataset2 for mae')
    test_model_1e4_f(model,x2,y2,f)
    print('\n')


# In[ ]:




