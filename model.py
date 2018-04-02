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
#######################################################################
#   THIS IS HERE TO PREVENT A TENSORFLOW ERROR
from keras import backend as K
K.set_image_dim_ordering('th')
#######################################################################
#   nist.gov/itl/iad/image-group/emnist-dataset
#data = io.loadmat("./emnist-letters.mat")
#######################################################################
class load_model ():
    
    def __init__(self):
        self.data = io.loadmat("./emnist-letters.mat")
        # train labels and data
        self.x_train = self.data["dataset"][0][0][0][0][0][0].astype(np.float32)
        self.train_labels = self.data["dataset"][0][0][0][0][0][1]
        # testing labels and data
        self.x_test = self.data["dataset"][0][0][1][0][0][0].astype(np.float32)
        self.test_labels = self.data["dataset"][0][0][1][0][0][1]
        self.model = Sequential()
        print("model initialized")
    def mod_data (self):
        self.x_train /=  255
        self.x_test /= 255

        # reshape vector
        self.x_train  = self.x_train.reshape(self.x_train.shape[0], 1, 28, 28, order="A")
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, 28, 28, order="A")

        # this fixes an error 

        self.train_labels = self.train_labels - 1
        self.test_labels = self.test_labels - 1 
        print('data moded')
    
    def normalize (self, x):
        print('normalized')
        return (x - self.x_train.mean().astype(np.float32)) / self.x_train.std().astype(np.float32)
    
    def modelize (self):
        # this will abstract the MNIST for experts
        # this will go through and normalize our inputs using built in Lambda
        self.model.add(Lambda(self.normalize,  input_shape=(1,28,28), output_shape=(1,28,28)))
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
        print('modelized')

    def compile (self) :
        self.model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        print("compiled")
    def img_process (self) :
        pass



        
mod = load_model()
mod.mod_data()
mod.modelize()
mod.compile()
