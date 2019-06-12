from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# graph to look at full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


print(tf.__version__)

# import Fashion MNIST dataset from Keras
fashion_mnist = keras.datasets.fashion_mnist
# Load training and test sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# images arrays are 28x28 NumPy arrays with pixel values from 0 to 255
# labels arrays are integer arrays ranging from 0 - 9 with the following representation
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Exploring the format of the data
print('Training Data')
#   60,000 images of size 28x28
print(train_images.shape)
#   60,000 labels in the training set
print(len(train_labels))
#   each label is an integer between 0 and 9
print(train_labels)
print('')

print('Testing Data')
#   10,000 images in the test set
print(test_images.shape)
#   10,000 image layers
print(len(test_labels))
print('')

'''
# Preprocess the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# Scale them to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
# verify data is in correct format
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # transforms 2D array of 28x28 to 1D array of 784 pixels
    keras.layers.Dense(128, activation='relu'), # creates densely (fully) connected neural layers with 128 neurons
    keras.layers.Dense(10, activation='softmax') # 10-node softmax output layer of 10 probability scores
])

# Compile the model
model.compile(optimizer='adam', # how the model is updated based on the data it sees and its loss function
              loss='sparse_categorical_crossentropy', # measures how accurate model is during training
              metrics=['accuracy']) # monitor training and testing steps (accuracy is fraction correctly identified)

# Adam: computes learning rates dynamically for each parameter
# Sparse Categorical Crossentropy: This is used when your labels are encoded as integers rather than one-hot
#   One-hot would use Categorical Crossentropy (not sparse)

# Train the model
#   Feed the training data to the model
#   Model learns to associate which images and labels
#   Ask it to make predictions about our test set
model.fit(train_images, train_labels, epochs=10)

# Evaluate Accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))

# So basically we overfit to our training data. This means we got too specific to our training data
# and we perform worse on any new, unseen data

# Make predictions
predictions = model.predict(test_images)
# This lets the model predict the label for each image in the testing set
# It's an array of 10 numbers which represent the model's confidence that
# the image corresponds to each of the 10 different articles of clothing

# We defined some great functions above to graph full set of predictions
'''
# This code lets us look at the 0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

# this looks at the 12th
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()
'''

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Now we can use the trained model to make a prediction about a single image
img = test_images[0]
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))
