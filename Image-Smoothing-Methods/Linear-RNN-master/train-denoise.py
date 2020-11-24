import numpy as np
import keras.keras as Keras
import skimage
import skimage.io
from skimage.util import random_noise, dtype
import matplotlib.pyplot as plt
import os

from LRNN import gen_model
from data.gen_patch import gen_patch

def datagen(pathnames, patchsize, batch_size):
    images = np.empty((batch_size, patchsize, patchsize, 3), dtype=np.float64)
    noisy_images = np.empty((batch_size, patchsize, patchsize, 3), dtype=np.float64)
    batch_size = batch_size // 4
    while True:
        np.random.shuffle(pathnames)
        for i in range(0, len(pathnames), batch_size):
            n_sample = 0
            for f in pathnames[i:i+batch_size]:
                img = skimage.io.imread(f)
                for _ in range(4):
                    patch = gen_patch(img, patchsize)
                    patch = skimage.util.img_as_float(patch)
                    if patch.ndim == 2:
                        patch = skimage.color.gray2rgb(patch)
                    images[n_sample] = patch
                    gauss_var = (0.01 + 0.09 * np.random.random()) ** 2
                    noisy_images[n_sample] = random_noise(patch, var=gauss_var)
                    n_sample = n_sample + 1
            yield (noisy_images[:n_sample], images[:n_sample])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0x5EED)

model = gen_model((96, 96, 3), 'parallel')
model.compile(optimizer='Adamax', loss=Keras.losses.mse)
# Keras.utils.plot_model(model, to_file='model.png')

batch_size = 32
patchsize = 96
n_epoch = 30

datadir = './data/train2014'
pathnames = [os.path.join(datadir, f) for f in os.listdir(datadir)]
n_sample = len(pathnames)
steps_per_epoch=int((n_sample + batch_size-1) / batch_size)
steps_per_epoch=100
n_epoch=300000

valdir = './data/test2014'
valnames = [os.path.join(valdir, f) for f in os.listdir(valdir)]

outputdir = './model-denoise'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

for epoch in range(n_epoch):
    model.fit_generator(
        datagen(pathnames, patchsize, batch_size),
        steps_per_epoch=steps_per_epoch,
        max_q_size=512,
        validation_data=datagen(valnames, patchsize, batch_size),
        validation_steps=10,
        initial_epoch=epoch,
        epochs=epoch+1
    )
    model.save_weights(
        os.path.join(outputdir, 'parallel-denoise-weights-%d.h5' % (epoch+1))
    )
