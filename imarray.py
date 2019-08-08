import os
import numpy as np
from PIL import Image


def image_to_array(directory, name):
    image = Image.open(directory + name)
    imgSize = image.size
    rawData = image.tobytes()
    img = Image.frombytes('RGB', imgSize, rawData)
    return img


numpy_output_data = []
path = 'env/test_clean/test_resized_final/'
test_files = sorted(os.listdir(path))[:30000]  # [:number from resize.py]
for img in test_files:
    im = Image.open(os.path.join(path, img))
    np_im = np.array(im)
    numpy_output_data.append(np_im)

numpy_output_data = np.array(numpy_output_data).reshape(-1, 64, 64, 3)
print(numpy_output_data.shape)
np.save('test_clean', numpy_output_data) #newtheme_nature_eval
