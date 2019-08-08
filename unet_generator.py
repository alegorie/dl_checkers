import keras.layers as KL
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import random
from modify_dataset import modify_dataset, modify_val

X_train = np.load('numpy_output.npy')[:10000]
y_train = np.load('checkers_10k.npy')[:10000]

# Data normalization. better do here (after test_train_split) of MEMORY ERROR!!!!!!
X_train = X_train.astype('float32')


def generator(features, labels, batch_size=100):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 24))

    while True:
        for i in range(batch_size):
            index = random.choice(range(1, len(features)))
            batch_features[i] = modify_dataset(features[index])
            batch_labels[i] = labels[index]

        yield batch_features, batch_labels


def val_generator(features, labels, batch_size=100):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 24))

    while True:
        for i in range(batch_size):
            index = random.choice(range(1, len(features)))
            batch_features[i] = modify_val(features[index])
            batch_labels[i] = labels[index]

        yield batch_features, batch_labels


# x, y = next(generator(X_train, y_train, 100))

input_layer = KL.Input((64, 64, 3))

conv1 = KL.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_layer)
conv1 = KL.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
pool1 = KL.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = KL.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
conv2 = KL.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
pool2 = KL.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = KL.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
conv3 = KL.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
pool3 = KL.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = KL.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
conv4 = KL.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
pool4 = KL.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = KL.Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
conv5 = KL.Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

up6 = KL.concatenate([KL.UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
conv6 = KL.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
conv6 = KL.Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

up7 = KL.concatenate([KL.UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
conv7 = KL.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
conv7 = KL.Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

up8 = KL.concatenate([KL.UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
conv8 = KL.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
conv8 = KL.Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

up9 = KL.concatenate([KL.UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
conv9 = KL.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
conv9 = KL.Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

flatten = KL.Flatten()(conv9)
dense = KL.Dense(24, activation='linear')(flatten)

model = Model(input=input_layer, output=dense)

X_train = np.load('numpy_output.npy')
Y_train = np.load('checkers_40k.npy')

# Data normalization. better do here (after test_train_split) of MEMORY ERROR!!!!!!
X_train = X_train.astype('float32')
# opt = SGD(lr=0.0003, momentum=0.9, decay=0.01)


opt = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mse', optimizer=opt, metrics=['mae'])

x_train = X_train.copy()
X_train = x_train[:8000]
y_train = Y_train[:8000]
X_test = x_train[8000:10000]
y_test = Y_train[8000:10000]

model.fit_generator(generator(X_train, y_train, 32),
                    validation_data=val_generator(X_test, y_test, 32),
                    steps_per_epoch=8000 // 32,
                    validation_steps=2000 // 32,
                    samples_per_epoch=8000 // 32,
                    nb_epoch=50,
                    verbose=1)

model.save('checkers_counter_generator.h5')


# model.fit(X_train, y_train,
#           validation_split=0.2,
#           batch_size=100,
#           epochs=100,
#           shuffle=True)

# results = model.predict(X_train)
# results = [[round(val) for val in sublst] for sublst in results]
# results = np.array(results).astype('int32')
# y_train = np.array(y_train).astype('int32')

# path = 'env/NN_predictions/'


def resave_images(directory, imarray, answerarray):
    import PIL
    from PIL import Image
    img = Image.fromarray(imarray, 'RGB')
    img.save(directory + str(answerarray) + '.png')

# X_train = np.load('numpy_output.npy')

# for i in range(10000):
#     resave_images(path, X_train[i], results[i])

# creates a HDF5 file 'model_name.h5'
# del model  # deletes the existing model
# print(model.evaluate(X_train, y_train))
