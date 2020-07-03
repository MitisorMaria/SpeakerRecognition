#!/usr/bin/env python
# coding: utf-8

# #### Importăm modulele necesare

# In[1]:


import signal_processing, data_load, learn, evaluate, render


# #### Valorile parametrilor pentru extragerea datelor

# In[2]:


num_rows_mfcc = 45
num_cols_mfcc = 130
num_rows_melspec = 32
num_cols_melspec = 110
num_channels = 1
num_labels = 10

num_epochs = 100
num_batch_size = 13
num_speakers = 1211
num_seconds = 3.0
n_fft = 2048
hop_length = 1024


# #### Creăm modelul 

# In[3]:


model = learn.create_model_with_two_inputs(num_labels, num_rows_mfcc, num_cols_mfcc, num_rows_melspec, num_cols_melspec, num_channels)


# #### Compilăm modelul

# In[4]:


learn.compile(model)


# #### Creăm dataframe-urile cu feature-urile obținute (două dataframes, unul cu mfcc, celălalt cu melspectrogram

# In[5]:


featuresdf1 = data_load.make_dataframe_class_no(num_speakers, './Dataset/wav/', num_rows_mfcc, num_cols_mfcc, num_seconds)


# In[6]:


featuresdf2 = data_load.make_dataframe_melspectrogram(num_speakers, './Dataset/wav/', num_rows_melspec, n_fft, hop_length, num_cols_melspec, num_seconds)


# #### Antrenăm modelul

# In[7]:


class_weights = learn.calculate_class_weight(featuresdf1)
history = learn.train_model_two_inputs(model, featuresdf1, featuresdf2, num_rows_mfcc, num_cols_mfcc, num_channels, num_rows_melspec, num_cols_melspec, num_epochs, num_batch_size, 
                                        'CNN1211', 'default', 'CNN1211', class_weights)


# In[24]:


result_sets_mfcc = data_load.make_train_test_sets(featuresdf1, num_rows_mfcc, num_cols_mfcc, num_channels)
result_sets_melspec = data_load.make_train_test_sets(featuresdf2, num_rows_melspec, num_cols_melspec, num_channels)

evaluate.evaluate_model_two_inputs(model, 'CNN1211', result_sets_mfcc, result_sets_melspec)



evaluate.display_metrics_two_inputs(model, 'CNN1211', result_sets_mfcc, result_sets_melspec)






