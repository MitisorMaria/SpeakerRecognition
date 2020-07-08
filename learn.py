import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
import matplotlib.pyplot as plt
import pathlib
import os
from tensorflow.keras.callbacks import ModelCheckpoint 
from datetime import datetime
import data_load

def train_model_two_inputs(model, featuresdf1, featuresdf2, num_rows_mfcc, num_cols_mfcc, num_channels, num_rows_melspec, num_cols_melspec, num_epochs, num_batch_size, 
	model_name, name, logdir_name, class_weight):
	
	saved_models = pathlib.Path('saved_models/')
	if not saved_models.exists():
		os.makedirs(saved_models)

	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.' + model_name + '.hdf5', 
								   verbose=1, save_best_only=True)
								   
	
	result_sets_mfcc = data_load.make_train_test_sets(featuresdf1, num_rows_mfcc, num_cols_mfcc, num_channels)
	x_train_mfcc = result_sets_mfcc[0]
	y_train_mfcc =  result_sets_mfcc[2]
	x_test_mfcc = result_sets_mfcc[1]
	y_test_mfcc =  result_sets_mfcc[3]

	num_labels = result_sets_mfcc[4]
	result_sets_melspec = data_load.make_train_test_sets(featuresdf2, num_rows_melspec, num_cols_melspec, num_channels)
	x_train_melspec = result_sets_melspec[0]
	y_train_melspec =  result_sets_melspec[2]
	x_test_melspec = result_sets_melspec[1]
	y_test_melspec =  result_sets_melspec[3]
	
	logdir = make_logdir(logdir_name)
	
	model.fit(
    x=[x_train_mfcc, x_train_melspec], y=y_train_mfcc,
    validation_data=([x_test_mfcc, x_test_melspec], y_test_mfcc),
    epochs=num_epochs, batch_size=num_batch_size, 
	class_weight = class_weight,
	callbacks=[checkpointer, tf.keras.callbacks.TensorBoard(logdir/name)], verbose=1
	)

def create_model_with_two_inputs(num_labels, num_rows_mfcc, num_cols_mfcc, num_rows_melspec, num_cols_melspec, num_channels):
	mfccs = tensorflow.keras.Input(shape=(num_rows_mfcc, num_cols_mfcc, num_channels), name="mfccs")
	melspectrograms = tensorflow.keras.Input(shape=(num_rows_melspec, num_cols_melspec, num_channels), name="melspectrograms")

	x1 = tensorflow.keras.layers.Conv2D(filters=16, kernel_size=2, activation='relu')(mfccs)
	x1 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x1)
	x1 = tensorflow.keras.layers.Dropout(0.2)(x1)
	x1 = tensorflow.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu')(x1)
	x1 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x1)
	x1 = tensorflow.keras.layers.Dropout(0.2)(x1)
	x1 = tensorflow.keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu')(x1)
	x1 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x1)
	x1 = tensorflow.keras.layers.Dropout(0.2)(x1)
	x1 = tensorflow.keras.layers.Conv2D(filters=128, kernel_size=2, activation='relu')(x1)
	x1 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x1)
	x1 = tensorflow.keras.layers.Dropout(0.2)(x1)
	x1 = tensorflow.keras.layers.GlobalAveragePooling2D()(x1)

	x2 = tensorflow.keras.layers.Conv2D(filters=16, kernel_size=2, activation='relu')(melspectrograms)
	x2 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x2)
	x2 = tensorflow.keras.layers.Dropout(0.2)(x2)
	x2 = tensorflow.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu')(x2)
	x2 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x2)
	x2 = tensorflow.keras.layers.Dropout(0.2)(x2)
	x2 = tensorflow.keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu')(x2)
	x2 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x2)
	x2 = tensorflow.keras.layers.Dropout(0.2)(x2)
	x2 = tensorflow.keras.layers.Conv2D(filters=128, kernel_size=2, activation='relu')(x2)
	x2 = tensorflow.keras.layers.MaxPooling2D(pool_size=2)(x2)
	x2 = tensorflow.keras.layers.Dropout(0.2)(x2)
	x2 = tensorflow.keras.layers.GlobalAveragePooling2D()(x2)

	x = tensorflow.keras.layers.concatenate([x1, x2])

	class_output = tensorflow.keras.layers.Dense(num_labels, activation="softmax", name="class_output")(x)
	model = tensorflow.keras.Model(
		inputs=[mfccs, melspectrograms], outputs=[class_output]
	)

	return model


def get_densenet(num_rows, num_columns, num_channels, num_labels, fine_tune_at):
	# Create the base model from the pre-trained model MobileNet V2
	base_model = tf.keras.applications.DenseNet121(
		include_top=False, weights='imagenet', input_tensor=None, input_shape=(num_rows, num_columns, num_channels),
		pooling='avg', classes=num_labels
	)
	
	base_model.trainable = True

	# Let's take a look to see how many layers are in the base model
	print("Number of layers in the base model: ", len(base_model.layers))

	# Freeze all the layers before the `fine_tune_at` layer

	for layer in base_model.layers[:fine_tune_at]:
		layer.trainable =  False

	# Let's take a look at the base model architecture
	base_model.summary()
	return base_model
	

def build_model_densenet(base_model, num_labels):
	# Construct model 
	model = tf.keras.Sequential([
	  base_model,
	  Dense(num_labels, activation='softmax')
	])

	model.summary()
	return model

	

def get_mobilenet(num_rows, num_columns, num_channels, fine_tune_at):
	# Create the base model from the pre-trained model MobileNet V2
	base_model = tf.keras.applications.MobileNetV2(input_shape=(num_rows, num_columns, num_channels),
												  include_top=False,
												  weights='imagenet')
	base_model.trainable = True

	# Let's take a look to see how many layers are in the base model
	print("Number of layers in the base model: ", len(base_model.layers))

	# Freeze all the layers before the `fine_tune_at` layer

	for layer in base_model.layers[:fine_tune_at]:
		layer.trainable =  False

	# Let's take a look at the base model architecture
	base_model.summary()
	return base_model
	

def build_model_mobilenet(base_model, num_labels):
	# Construct model 
	model = tf.keras.Sequential([
	  base_model,
	  tf.keras.layers.GlobalAveragePooling2D(),
	  Dense(num_labels, activation='softmax')
	])
	model.summary()
	return model
	
def compile_model_pretrained_net(model, base_learning_rate):
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])


def build_CNN_model(num_rows, num_columns, num_labels):
	num_channels = 1
	filter_size = 2

	 # Construct model 
	model = Sequential()
	model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Dropout(0.2))

	model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Dropout(0.2))
	model.add(GlobalAveragePooling2D())

	model.add(Dense(num_labels, activation='softmax')) 
	return model
	

def build_regularized_CNN_model(num_rows, num_columns, num_labels):
	#Regularization:
	# Construct model
	num_channels = 1	
	modelReg = Sequential()
	modelReg.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
	modelReg.add(MaxPooling2D(pool_size=2))
	modelReg.add(Dropout(0.2))

	modelReg.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
	modelReg.add(MaxPooling2D(pool_size=2))
	modelReg.add(Dropout(0.2))

	modelReg.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
	modelReg.add(MaxPooling2D(pool_size=2))
	modelReg.add(Dropout(0.2))

	modelReg.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
	modelReg.add(MaxPooling2D(pool_size=2))
	modelReg.add(Dropout(0.2))
	modelReg.add(GlobalAveragePooling2D())

	modelReg.add(tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.001)))

	modelReg.add(Dense(num_labels, activation='softmax'))
	return modelReg


def compile(model):
	# Compile the model
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	

def make_logdir(dirname):
	logdir = pathlib.Path("./tensorboard_logs/" + dirname)
	if not logdir.exists():
		os.makedirs(logdir)
	name='default'
	return logdir
	

def calculate_class_weight(featuresdf):
	true_0, true_1, true_2, true_3, true_4, true_5, true_6, true_7, true_8, true_9 = np.bincount(featuresdf['class'])
	total = true_0+ true_1+ true_2+ true_3+ true_4+ true_5+ true_6+ true_7+ true_8+ true_9
	weight_for_0 = (1 / true_0)*(total)/2.0 
	weight_for_1 = (1 / true_1)*(total)/2.0
	weight_for_2 = (1 / true_2)*(total)/2.0 
	weight_for_3 = (1 / true_3)*(total)/2.0
	weight_for_4 = (1 / true_4)*(total)/2.0 
	weight_for_5 = (1 / true_5)*(total)/2.0
	weight_for_6 = (1 / true_6)*(total)/2.0 
	weight_for_7 = (1 / true_7)*(total)/2.0
	weight_for_8 = (1 / true_8)*(total)/2.0 
	weight_for_9 = (1 / true_9)*(total)/2.0

	class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3, 4: weight_for_4, 5: weight_for_5,
				6: weight_for_6, 7: weight_for_7, 8: weight_for_8, 9: weight_for_9}
	return class_weight


				   
def train_model_class_weights(model, train_test_data, num_epochs, num_batch_size, model_name, name, logdir_name, class_weight):
	saved_models = pathlib.Path('saved_models/')
	if not saved_models.exists():
		os.makedirs(saved_models)

	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.' + model_name + '.hdf5', 
								   verbose=1, save_best_only=True)
								   
	x_train = train_test_data[0]
	x_test = train_test_data[1]
	y_train = train_test_data[2]
	y_test = train_test_data[3]
	
	start = datetime.now()
	
	logdir = make_logdir(logdir_name)

	history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer, tf.keras.callbacks.TensorBoard(logdir/name)], verbose=1,
                   class_weight=class_weight)
	duration = datetime.now() - start
	print("Training completed in time: ", duration)
	return history

	
def train_model(model, train_test_data, num_epochs, num_batch_size, model_name, name, logdir_name):
	saved_models = pathlib.Path('saved_models/')
	if not saved_models.exists():
		os.makedirs(saved_models)

	checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.' + model_name + '.hdf5', 
								   verbose=1, save_best_only=True)
								   
	x_train = train_test_data[0]
	x_test = train_test_data[1]
	y_train = train_test_data[2]
	y_test = train_test_data[3]
	
	start = datetime.now()
	
	logdir = make_logdir(logdir_name)

	history = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer, tf.keras.callbacks.TensorBoard(logdir/name)], verbose=1)
	duration = datetime.now() - start
	print("Training completed in time: ", duration)
	return history