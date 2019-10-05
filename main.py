from keras import Input, Model
from keras.layers import Conv2D, Dense, BatchNormalization, Flatten, Activation, LeakyReLU
from keras.datasets import cifar10
from keras.utils import to_categorical
import pickle
import os

#set constants
NUM_CLASSES = 10

#pickle train data 
if os.path.exists("./pickles/cifarData") == False:
    #get cifar data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    cifardata = [x_train, y_train, x_test, y_test]
    #create file
    with open("./pickles/cifarData", 'wb+') as f:
        pickle.dump(cifardata, f)

else:
    with open("./pickles/cifarData", 'rb+') as f:
        cifardata = pickle.load(f)
    x_train, y_train, x_test, y_test = cifardata[0], cifardata[1], cifardata[2], cifardata[3]


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#create functional layers
input_layer = Input(shape=(32,32,3))
#input shape (32, 32, 3)
x = Conv2D(32, 3, strides=1, padding='same')(input_layer)
print(x.shape)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

#input shape (32, 32, 10)
x = Conv2D(32, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
print(x.shape)

#input shape(16,16,20)
x = Conv2D(64, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

#input shape(16,16, 10)
x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization(momentum=.9)(x)
x = LeakyReLU()(x)

print("last layer ", x.shape)
#flatten data before dense layer
x = Flatten()(x)
x = Dense(128)(x)

x = Dense(NUM_CLASSES)(x)
print("last layer layer ", x.shape)
output_layer = Activation('softmax')(x)


model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train)



