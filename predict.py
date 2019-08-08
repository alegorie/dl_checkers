from keras.models import load_model
import numpy as np
import h5py


X_test = h5py.File('checkers_data_storage.h5', 'r')['X_train'][:]
y_test = h5py.File('checkers_data_storage.h5', 'r')['y_train'][:]
# y_test = np.load('test_clean_checkers.npy')  # checkers_10k.npy checkers_30k numpy_output_test newtheme_nature_test
# X_test = np.load('test_clean.npy')  # numpy_output.npy numpy_output_test newtheme_nature_eval
X_test = X_test.astype('float32')
X_test /= 255

model = load_model('checkers_counter_generator.h5')

prediction = model.predict(X_test)
prediction = [[round(val) for val in sublst] for sublst in prediction]
prediction = np.array(prediction).astype('int32')

count = 0

for i in range(100, 200, 20):
    print(prediction[i])
    print(y_test[i])

l = []

for i in range(len(y_test)):
    if np.array_equal(prediction[i], y_test[i]):
        count += 1
        l.append(i)

print(len(l))
print("{:.2%}".format(count / len(y_test)))
