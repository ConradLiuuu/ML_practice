import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    #normalization
    x_train = x_train / 255
    x_test = x_test /255
    # add noise
    #x_test = np.random.normal(x_test)
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()
# input layer
model.add(Dense(input_dim=28*28, units=512, activation='sigmoid'))
#model.add(Dropout(0.5)) # when noise comes can try droout
# hidden layer
model.add(Dense(units=512, activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(units=512, activation='sigmoid'))
#model.add(Dropout(0.5))
#for i in range(10):
    #model.add(Dense(units=512, activation='sigmoid'))
# output layer
model.add(Dense(units=10, activation='softmax'))

#model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20)

result = model.evaluate(x_train, y_train, batch_size=10000)
print ('\nTraining accuracy:', result[1])

result = model.evaluate(x_test, y_test, batch_size=10000)
# result[0]=Total loss, result[1]=Accuracy
print ('\nTest accuracy:', result[1])

