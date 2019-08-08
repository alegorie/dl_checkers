import numpy as np
import random
from cvutil import cvslack
import os

from keras.models import load_model
import numpy as np

X_test = np.load('test_clean.npy')  # numpy_output.npy numpy_output_test newtheme_nature_eval
X_test = X_test.astype('float32')
X_test /= 255
model = load_model('checkers_counter_generator.h5')
prediction = model.predict(X_test)
prediction = [[round(val) for val in sublst] for sublst in prediction]
prediction = np.array(prediction).astype('int32')
y_test = np.load('test_clean_checkers.npy')


def p_image(directory, name, ind, x, y):  # post 1 image
    from PIL import Image

    img = Image.open(directory + name)
    cvslack.post_image(np.array(img), str(y[i])+str(x[i])+'.png', 'flow-ymarkov')


path = 'env/test_clean/test_resized_final/'  # folder with images to post

files = os.listdir(path)
index = random.randrange(0, len(files))

for i in range(50, 60, 2):
    print(files[i])
    p_image(path, files[i], i, prediction, y_test)

# for i in range(3):
# 	rand = random.randint(0, 9999)
# 	im = arr[rand]
# 	cvslack.post_image(im, str(rand)+'.png', 'flow-ymarkov')
