###########################TENSOR_FLOW####################################
import tensorflow as tf
sess = tf.InteractiveSession()
##########################################################################
############################OTHER_TOOLS###################################
from scipy import io
import numpy as np
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
#   THIS IS HERE TO PREVENT A TENSORFLOW ERROR
from keras import backend as K
K.set_image_dim_ordering('th')
#######################################################################
#   nist.gov/itl/iad/image-group/emnist-dataset
#datas = io.loadmat("./emnist-letters.mat")
#######################################################################
data = io.loadmat("./emnist-letters.mat")
# train labels and data
x_train = data["dataset"][0][0][0][0][0][0].astype(np.float32)
train_labels = data["dataset"][0][0][0][0][0][1]
# testing labels and data
x_test = data["dataset"][0][0][1][0][0][0].astype(np.float32)
test_labels = data["dataset"][0][0][1][0][0][1]

x_train /=  255
x_test /= 255      
# reshape vector
x_train  = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")
# this fixes an error 
train_labels = train_labels - 1
test_labels = test_labels - 1


def normalize (x):
    # print('normalized')
    return (x - x_train.mean().astype(np.float32)) / x_train.std().astype(np.float32)


class load_model ():

    def __init__ (self):
        self.model = Sequential()
        self.modelize()
        self.compile()

    def modelize (self):
        # this will abstract the MNIST for experts
        # this will go through and normalize our inputs using built in Lambda
        self.model.add(Lambda(normalize,  input_shape=(1,28,28), output_shape=(1,28,28)))
        # 1st layer convolution 
        self.model.add(Conv2D(32, (5,5)))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization(axis = 1))
        # 2nd layer convolution
        self.model.add(Conv2D(32, (5,5)))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization(axis = 1))       
        # more fitting 
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(512))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(26, activation='softmax'))
        # print('modelized')

    def compile (self) :
        self.model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        # print("compiled")

    def data_augment (self) :
        pass

        
models = []
weights_epoch = 0


for i in range(10):
    m = load_model()
    models.append(m)

eval_batch_size = 100
all_preds = np.stack([m.model.predict(x_test, batch_size=eval_batch_size) for m in models])
avg_preds = all_preds.mean(axis=0)
print((1 - keras.metrics.categorical_accuracy(test_labels, avg_preds).eval().mean()) * 100)


