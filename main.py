from keras import Input
from keras.layers import Conv2D, Dense, BatchNormalization
from keras.datasets import cifar10


#set constants
num_classes = 10


#get cifar data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#create functional layers
x = Input((32,32,3))
x = Conv2D(10, (3,3), 1)(x)
x = BatchNormalization(momentum=.9)


