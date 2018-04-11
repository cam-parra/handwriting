###########################TENSOR_FLOW####################################
import tensorflow as tf

sess = tf.InteractiveSession()
##########################################################################
############################OTHER_TOOLS###################################
from scipy import io
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###########################KERAS_API#####################################
#   layers:
#       -   Dense
#       -   Dropout
#       -   Conv2D
#       -   MaxPooling2D
#       -   Flatten
#       -   Lambda
#       -   BatchNormalization
#       -   LeakyReLU
#   Optimizers:
#       -   Adam
#######################################################################
#######################################################################
import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
#######################################################################
#   THIS IS HERE TO PREVENT A TENSORFLOW ERROR might not be needed on different systems
from keras import backend as K

K.set_image_dim_ordering('th')
#######################################################################
#   nist.gov/itl/iad/image-group/emnist-dataset
# datas = io.loadmat("./emnist-letters.mat")
#######################################################################
batch_size = 512
print("\n\n\n#######################################################################")
print("\n\nWelcome to the handwriting recognition project")
print("\n\n\n#######################################################################")
input_user = input("1. Letters 2. Numbers \n~----------> ")

if input_user == 1:
    data = io.loadmat("./emnist-letters.mat")
elif input_user == 2:
    data = io.loadmat("./emist-mnist.mat")
else:
    print("\n\n~~~~~~~~~~ Learning Resuming ~~~~~~~~~~\n ")
    data = io.loadmat("./emnist-letters.mat")

# train labels and data
train = data["dataset"][0][0][0][0][0][0].astype(np.float32)
train_labels = data["dataset"][0][0][0][0][0][1]
# testing labels and data
test = data["dataset"][0][0][1][0][0][0].astype(np.float32)
test_labels = data["dataset"][0][0][1][0][0][1]

train /= 255
test /= 255
# reshape vector
train = train.reshape(train.shape[0], 1, 28, 28, order="A")
test = test.reshape(test.shape[0], 1, 28, 28, order="A")
# this fixes an error
test_labels = test_labels - 1
train_labels = train_labels - 1

train_labels = keras.utils.to_categorical(train_labels, 26)
test_labels = keras.utils.to_categorical(test_labels, 26)

visu_one = input("Would you like to see a sample of your data? [Y/n]\n~----------> ")
if visu_one == 'Y':
    num = int(input("Please pick a number from 0 - 10000\n~----------> "))
    import matplotlib.pyplot as plot

    sample_image = train[num]
    plot.imshow(sample_image[0], cmap='gray')
    plot.show()
    print(train_labels[num][0])
elif visu_one != 'Y':
    print("\n~~~~~~~~~~ Learning Resuming ~~~~~~~~~~\n")

from keras.preprocessing.image import ImageDataGenerator as idg
conditions = {'rotation_range': 12, 'width_shift_range': 0.1,
                  'shear_range': 0.3, 'height_shift_range': 0.1,
                  'zoom_range': 0.1, 'data_format': 'channels_first'}
aug_gen = idg(**conditions)
all_batches = aug_gen.flow(train, train_labels, batch_size=batch_size)
testing = aug_gen.flow(test, test_labels, batch_size=batch_size)
steps = int(np.ceil(all_batches.n / batch_size))
val_steps = int(np.ceil(testing.n/batch_size))


############################################################
# func: normalize 
#
############################################################
def normalize(x):
    # print('normalized')
    return (x - train.mean().astype(np.float32)) / train.std().astype(np.float32)


class load_model():

    def __init__(self):
        self.model = Sequential()
        self.modelize()
        self.compile()

    def modelize(self):
        # this will abstract the MNIST for experts
        # this will go through and normalize our inputs using built in Lambda
        self.model.add(Lambda(normalize, input_shape=(1, 28, 28), output_shape=(1, 28, 28)))
        # 1st layer convolution
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization(axis=1))
        # 2nd layer convolution
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization(axis=1))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization(axis=1))
        # more fitting

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(512))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(26, activation='softmax'))
        # print('modelized')

    def compile(self):
        self.model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        # print("compiled")


models = []
weights_epoch = 4
# loads our model from past experiments
for i in range(10):
    m = load_model()
    models.append(m)
models = []

for i in range(10):
    m = load_model()
    m.model.load_weights('learning/weights/{:03d}epochs_weights_model_{}.pkl'.format(weights_epoch, i))
    models.append(m)
num_it = 1
total = 1

eval_batch_size = 512


# for it in range(num_it):
#     epoch = (it + 1)* total + weights_epoch
#     print('\ncurrent iteration: ' + str(it) + '\ncurrent epoch: ' + str(epoch) +'\n')
#
#     for index,cur_model in enumerate(models):
#         print('I am in the second loop')
#         cur_model.model.optimizer.lr = 0.000001
#
#         trained = cur_model.model.fit_generator(all_batches,steps_per_epoch=steps, epochs=total, verbose=0,
#                             validation_data=testing, validation_steps=val_steps)
#         cur_model.model.save_weights("learning/weights/{:03d}epochs_weights_model_{}.pkl".format(epoch, index))
#     all_preds = np.stack([m.model.predict(test, batch_size=eval_batch_size) for m in models])
#     avg_preds = all_preds.mean(axis=0)
#     test_error = (1 - keras.metrics.categorical_accuracy(test_labels, avg_preds).eval().mean()) * 100
#
#     with open("learning/history/test_errors_epoch_{:03d}.txt".format(epoch), "w") as text_file:
#         print('i am in the third loop')
#         text_file.write("epoch: {} test error on ensemble: {}\n".format(epoch, test_error))
#         for m in models:
#             pred = np.array(m.predict(test, batch_size=eval_batch_size))
#             test_err = (1 - keras.metrics.categorical_accuracy(test_labels, pred).eval().mean()) * 100
#             text_file.write("{}\n".format(test_err))
#

# for presentation only use above code to really ensemble learn and get best prediction
# from keras.utils import plot_model
# plot_model(models[0].model, to_file='conv_model.png')
# exit()

models[0].model.optimizer.lr = 0.0001
history = models[0].model.fit_generator(all_batches, steps_per_epoch=steps, epochs=4,
                   validation_data=testing, validation_steps=val_steps)
pred = np.array(models[0].model.predict(test, batch_size=eval_batch_size))
print(0, (1 - keras.metrics.categorical_accuracy(test_labels, pred).eval().mean()) * 100)


# all_preds = np.stack([m.model.predict(test, batch_size=eval_batch_size) for m in models])
# avg_preds = all_preds.mean(axis=0)
# print((1 - keras.metrics.categorical_accuracy(test_labels, avg_preds).eval().mean()) * 100)







