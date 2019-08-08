import os


def resize_image(directory, name):
    import PIL
    from PIL import Image
    size = 64, 64
    img = Image.open(directory + name)
    img = img.resize(size, PIL.Image.ANTIALIAS)
    img.save('env/test_clean/test_resized_final/' + name)

path = '/mnt/network-storage/bgdata/test_clean_checkers/'
print('start')
list_of_images = os.listdir(path)
for filename in sorted(list_of_images)[:30000]:
    if filename.endswith(".png"):
        resize_image(path, filename)

print(len(list_of_images))
