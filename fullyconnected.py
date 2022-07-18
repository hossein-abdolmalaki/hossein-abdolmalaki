#!pip install keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def plot_history(net_history):
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['training_loss', 'test_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['training_accuracy', 'test_accuracy'])

# Seed random number generator
random_state = 42
np.random.seed(random_state)


# Function to load training and test data
def load(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

path_file="/content/"
#path_file='/content/drive/My Drive/majd_proj/'
# Loading data
data_x=a3.copy()
data_y=b3.copy() 
#data_x, data_y = load(path_file+'train_spins0_rev'), load(path_file+'train_labels0_rev')
#, load('H_extenal_field')
#data_x = np.loadtxt(path_file+'train_spins0.txt')
#data_y = np.loadtxt(path_file+'train_labels0.txt')
# Split data into training and test set
#train_x, test_x, train_y, test_y, train_T, test_T = train_test_split(data_x, data_y, T, test_size=0.1, random_state=random_state)
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=random_state)

print(data_x[0])
print(data_y[0])
# Parameters
learning_rate = 1e-2
l2 = 2 * 1e-5
training_epochs = 5000
display_step = 100

# Network Parameters
n_hidden_1 = 100 # 1st layer number of features
n_input = 16 * 16 # 2D Ising lattice
n_classes = 4 # high and low phase

#==================================================
# Creating our model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import gradient_descent_v2
from keras.losses import categorical_crossentropy
sgd = gradient_descent_v2.SGD()
myModel = Sequential()
myModel.add(Dense(400, activation='relu', input_shape=(400,)))
myModel.add(Dropout(.2))
myModel.add(Dense(200, activation='relu'))
myModel.add(Dropout(.2))
myModel.add(Dense(4, activation='sigmoid'))

myModel.summary()
myModel.compile(optimizer=sgd, loss=categorical_crossentropy, metrics=['accuracy'])

#==================================================
# Train our model
network_history = myModel.fit(train_x, train_y, batch_size=64, epochs=200, validation_split=0.2)
plot_history(network_history)

# Evaluation
test_loss, test_acc = myModel.evaluate(test_x, test_y)
test_labels_p = myModel.predict(test_x)
import numpy as np
#test_labels_p = np.argmax(test_labels_p, axis=1)

# Change layers config
#myModel.layers[0].name = 'Layer_00'
#myModel.layers[0].trainable = False
#myModel.layers[0].get_config()

