#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# In[2]:


x_train = np.loadtxt('new_600k_x_train_2nd_set_only_Grid')
x_test = np.loadtxt('new_600k_x_test_2nd_set_only_Grid')
y_train = np.loadtxt('new_600k_y_train_2nd_set_only_Grid')
y_test = np.loadtxt('new_600k_y_test_2nd_set_only_Grid')


# In[3]:


print(x_train.shape, x_test.shape)


# In[4]:


x_combined = np.vstack((x_train, x_test))
y_combined = np.vstack((y_train, y_test))

kn = 2.0e8 
kt = 0.5 * kn
factors = np.array([kn*1e-4,kn*1e-4,kt*1e-4,1e4])

y_combined_multiplied = y_combined * factors

y_train_multiplied = y_combined_multiplied[:len(y_train)]
y_test_multiplied = y_combined_multiplied[len(y_train):]


# In[5]:


print(y_train_multiplied.shape)
print(y_test_multiplied.shape)
print(np.max(y_train_multiplied,axis = 0))


# In[6]:


np.savetxt('new_600k_y_train_multiplied.txt', y_train_multiplied)
np.savetxt('new_600k_y_test_multiplied.txt', y_test_multiplied)


# In[7]:


# Define normalization functions that also return the scaler
def min_max_scaling(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def z_score_standardization(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def robust_scaling(data):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def unit_vector_scaling(data):
    norms = np.linalg.norm(data, axis=1)
    scaled_data = data / norms[:, np.newaxis]
    return scaled_data

def max_absolute_scaling(data):
    max_abs = np.max(np.abs(data), axis=0)
    scaled_data = data / max_abs
    return scaled_data


# In[8]:


# Apply each normalization function to the combined input data
x_train_normalized_min_max, x_scaler_min_max = min_max_scaling(x_train)
x_test_normalized_min_max = x_scaler_min_max.transform(x_test)

x_train_standardized_z_score, x_scaler_z_score = z_score_standardization(x_train)
x_test_standardized_z_score = x_scaler_z_score.transform(x_test)

x_train_robust_scaled, x_scaler_robust = robust_scaling(x_train)
x_test_robust_scaled = x_scaler_robust.transform(x_test)

x_train_normalized_unit_vector = unit_vector_scaling(x_train)
x_test_normalized_unit_vector = unit_vector_scaling(x_test)

x_train_scaled_max_abs = max_absolute_scaling(x_train)
x_test_scaled_max_abs = max_absolute_scaling(x_test)


# In[9]:


# Save the scalers that need to be saved
with open('x_scaler_min_max.pkl', 'wb') as x_scaler_min_max_file:
    pickle.dump(x_scaler_min_max, x_scaler_min_max_file)

with open('x_scaler_z_score.pkl', 'wb') as x_scaler_z_score_file:
    pickle.dump(x_scaler_z_score, x_scaler_z_score_file)

with open('x_scaler_robust.pkl', 'wb') as x_scaler_robust_file:
    pickle.dump(x_scaler_robust, x_scaler_robust_file)

# Save normalized input training data
np.savetxt('x_train_normalized_min_max.txt', x_train_normalized_min_max)
np.savetxt('x_train_standardized_z_score.txt', x_train_standardized_z_score)
np.savetxt('x_train_robust_scaled.txt', x_train_robust_scaled)
np.savetxt('x_train_normalized_unit_vector.txt', x_train_normalized_unit_vector)
np.savetxt('x_train_scaled_max_abs.txt', x_train_scaled_max_abs)

# Save normalized input test data
np.savetxt('x_test_normalized_min_max.txt', x_test_normalized_min_max)
np.savetxt('x_test_standardized_z_score.txt', x_test_standardized_z_score)
np.savetxt('x_test_robust_scaled.txt', x_test_robust_scaled)
np.savetxt('x_test_normalized_unit_vector.txt', x_test_normalized_unit_vector)
np.savetxt('x_test_scaled_max_abs.txt', x_test_scaled_max_abs)


# In[ ]:




