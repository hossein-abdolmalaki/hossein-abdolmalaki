#!pip install keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def plot_history(net_history):
    history = net_history.history
    import matplotlib.pyplot as plt
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

# Loading data
#data_x, data_y = load('train_spins'), load('train_labels')
#T = load('H_extenal_field')
path_file='/content/drive/My Drive/majd_proj/'
# Loading data
data_x=a3.copy()
data_y=b3.copy() 
#data_x, data_y = load(path_file+'train_spins0'), load(path_file+'train_labels0')
#, load('H_extenal_field')
#data_x = np.loadtxt(path_file+'train_spins0.txt')
#data_y = np.loadtxt(path_file+'train_labels0.txt')
# Split data into training and test set
#train_x, test_x, train_y, test_y, train_T, test_T = train_test_split(data_x, data_y, T, test_size=0.1, random_state=random_state)
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=random_state)

train_x = train_x.reshape(len(train_x), 20, 20, 1)
test_x = test_x.reshape(len(test_x), 20, 20, 1)

print(train_x[0])
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

#==================================================
# Creating our model
from keras.models import Model
from keras import layers
import keras
from keras.optimizers import adam_v2

from keras.optimizers import gradient_descent_v2
from keras.losses import categorical_crossentropy
sgd = gradient_descent_v2.SGD()

myInput = layers.Input(shape=(20,20,1))
conv1 = layers.Conv2D(32, 2, activation='relu', padding='same', strides=4)(myInput)
conv2 = layers.Conv2D(64, 2, activation='relu', padding='same', strides=4)(conv1)
flat = layers.Flatten()(conv2)
dense1=layers.Dense(100, activation='relu')(flat)
Dropout1=layers.Dropout(.2)(dense1)
out_layer = layers.Dense(4, activation='sigmoid')(Dropout1)

myModel = Model(myInput, out_layer)

myModel.summary()
myModel.compile(optimizer=keras.optimizers.adam_experimental.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


#==================================================
# Train our model
network_history = myModel.fit(train_x, train_y, batch_size=128, epochs=500, validation_split=0.2)
plot_history(network_history)

# Evaluation
test_loss, test_acc = myModel.evaluate(test_x, test_y)
test_labels_p = myModel.predict(test_x)
import numpy as np
test_labels_p = np.argmax(test_labels_p, axis=1)

# Change layers config
#myModel.layers[0].name = 'Layer_00'
myModel.layers[0].trainable = False
myModel.layers[0].get_config()
