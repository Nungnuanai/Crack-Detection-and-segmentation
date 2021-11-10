from __future__ import print_function
import os
from scipy import misc
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
#from tensorflow.keras.constraints import maxnorm

from tensorflow.keras.optimizers import SGD

import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import image
import matplotlib.pyplot as plt

file_dirs = []
for root, dirs, files in os.walk('train', topdown=False):
    for name in files:
        file_dirs.append(os.path.join(root, name))


random.shuffle(file_dirs)
num_samples = len(file_dirs)
X = []
Y = []

for i in range(0, num_samples):
    f = misc.imread(file_dirs[i], mode="RGB")
    #f = misc.imresize(i, (100, 100))
    X.append(f)
    folders = file_dirs[i].split('/')
    label = 0 if folders[len(folders) - 2] == 'negative' else 1
    Y.append(label)

X = np.array(X)
Y = np.array(Y)
print('X.shape, Y.shape',X.shape, Y.shape)
num_validate = int(num_samples*0.2)
num_test = int(num_samples*0.2)
num_val_test = num_validate + num_test
(x_train, y_train) = (X[:num_samples - num_val_test], Y[:num_samples - num_val_test])
(x_test, y_test) = (X[num_samples - num_val_test:num_samples - num_test],
                    Y[num_samples - num_val_test:num_samples - num_test])

batch_size = 5
num_classes = 2
epochs = 2

learning_rate = 0.00001
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

img_rows, img_cols = 96,96

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

if keras.backend.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(12, kernel_size=(3, 3),
                 activation='relu', padding = 'same',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(18, kernel_size=(3, 3),
                 activation='relu', padding = 'same',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(24, kernel_size=(3, 3),
                 activation='relu', padding = 'same',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48, kernel_size=(3, 3),
                 activation='relu', padding = 'same',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu', padding = 'same',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
#    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=5
#    channel_shift_range=0.3
)
x_test /= 255
model.fit_generator(train_datagen.flow(x_train, y_train,
                                       batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

model.save('Test.h5')
model.summary()

x_val, y_val = (X[num_samples - num_test:], Y[num_samples - num_test:])
x_val = np.array(x_val)
y_val = np.array(y_val)
x_val = x_val.astype('float64')
#x_val = (x_val - mean) / (std + 1e-9)
x_val /= 255
prediction = model.predict(x_val)
y_val = keras.utils.to_categorical(y_val, num_classes)
acc_val = np.mean(np.round(prediction) == y_val)
print(acc_val)

keras.callbacks.History()
history = model.fit(x_test, y_test, epochs=epochs, validation_split=0.2, shuffle=True)

print(history.history.keys())
#  "Accuracy"
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Test model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
#plt.show()
plt.savefig('Testaccuracy_new_19_Dec.png')

# "Loss"
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig('Testloss_new_19_Dec.png')








