from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def evaluate_before_training(model, x_test, y_test):
	# Display model architecture summary 
	model.summary()

	# Calculate pre-training accuracy 
	score = model.evaluate(x_test, y_test, verbose=1)
	accuracy = 100*score[1]
	print("Pre-training accuracy: %.4f%%" % accuracy)
	
	
def evaluate_model_two_inputs(model, model_name, train_test_data1, train_test_data2):
	x_train1 = train_test_data1[0]
	x_test1 = train_test_data1[1]
	y_train1 = train_test_data1[2]
	y_test1 = train_test_data1[3]
	
	x_train2 = train_test_data2[0]
	x_test2 = train_test_data2[1]
	y_train2 = train_test_data2[2]
	y_test2 = train_test_data2[3]
	# Evaluating the model on the training and testing set
	model.load_weights("saved_models/weights.best." + model_name + ".hdf5")
	score = model.evaluate([x_train1, x_train2], y_train1, verbose=0)
	print("Training Accuracy: ", score[1])
	score = model.evaluate([x_test1, x_test2], y_test1, verbose=0)
	print("Testing Accuracy: ", score[1])
	
def display_metrics_two_inputs(model, model_name, train_test_data1, train_test_data2):
    x_train1 = train_test_data1[0]
    x_test1 = train_test_data1[1]
    y_train1 = train_test_data1[2]
    y_test1 = train_test_data1[3]
    
    x_train2 = train_test_data2[0]
    x_test2 = train_test_data2[1]
    y_train2 = train_test_data2[2]
    y_test2 = train_test_data2[3]
    
    model.load_weights("saved_models/weights.best." + model_name + ".hdf5")
    
    y_prob = model.predict([x_test1, x_test2]) 
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(y_test1, axis=1)

    print(classification_report(y_true, y_pred))
    print("Confusion matrix: ")
    print(confusion_matrix(y_true, y_pred))
	
def evaluate_model(model, model_name, train_test_data):
	x_train = train_test_data[0]
	x_test = train_test_data[1]
	y_train = train_test_data[2]
	y_test = train_test_data[3]
	# Evaluating the model on the training and testing set
	model.load_weights("saved_models/weights.best." + model_name + ".hdf5")
	score = model.evaluate(x_train, y_train, verbose=0)
	print("Training Accuracy: ", score[1])

	score = model.evaluate(x_test, y_test, verbose=0)
	print("Testing Accuracy: ", score[1])
	
def print_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
		
def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))#, sharey=True)
    #axs[0].bar(names, values)
    #axs[1].scatter(names, values)
    #axs[2].plot(names, values)

    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, '--', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc="upper right")

    axs[1].plot(epochs, acc, 'b', label='Training acc')
    axs[1].plot(epochs, val_acc, '--', label='Validation acc')
    axs[1].set_title('Training and validation acc')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc="upper right")

def plot_compared_val_loss(history1, history2, name1, name2):
    
    loss1 = history1.history['val_loss']
    loss2 = history2.history['val_loss']
    
    epochs = range(1, len(loss1) + 1)

    plt.plot(epochs, loss1, 'b', label=name1)
    plt.plot(epochs, loss2, '--', label=name2)
    plt.title('Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
	
def display_metrics(model, model_name, train_test_data):
	x_train = train_test_data[0]
	x_test = train_test_data[1]
	y_train = train_test_data[2]
	y_test = train_test_data[3]
	
	model.load_weights("saved_models/weights.best." + model_name + ".hdf5")
	
	y_pred = model.predict_classes(x_test, batch_size=12, verbose=0)
	y_true = np.argmax(y_test, axis=1)

	print(classification_report(y_true, y_pred))
	print("Confusion matrix: ")
	print(confusion_matrix(y_true, y_pred))