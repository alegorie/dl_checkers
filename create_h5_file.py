import numpy as np
import h5py


X_train = np.load('numpy_output.npy')
y_train = np.load('checkers_10k.npy')
X_test = np.load('test_clean.npy')  # numpy_output.npy numpy_output_test newtheme_nature_eval
y_test = np.load('test_clean_checkers.npy')  # checkers_10k.npy checkers_30k numpy_output_test newtheme_nature_test
X_nature = np.load('newtheme_nature_eval.npy')
y_nature = np.load('newtheme_nature_test.npy')

h5f = h5py.File('checkers_data_storage.h5', 'w')
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_nature', data=X_nature)
h5f.create_dataset('y_nature', data=y_nature)

h5f.close()