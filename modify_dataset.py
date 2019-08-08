import random as rand
# import matplotlib.pyplot as plt


def modify_dataset(X_train):
    inverted = X_train.copy()
    rand_chan = [rand.randint(1, 10), rand.randint(1, 10), rand.randint(1, 10)]
    rand_chan = set(rand_chan)
    for k in rand_chan:
        if k in [0, 1, 2]:
            # plt.imsave(str(rand_image)+'.png', inverted)
            inverted[:, :, k] = 255 - inverted[:, :, k]
            # inverted = np.array(inverted).reshape(-1, 64, 64, 3)

    return inverted / 255


def modify_val(X):
    return X / 255
