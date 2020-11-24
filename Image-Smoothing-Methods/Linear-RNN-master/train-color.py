import numpy as np
import keras.keras as Keras
import skimage
import skimage.io
from skimage.util import random_noise, dtype
from skimage.color import rgb2yuv
import matplotlib.pyplot as plt
import os
from sys import argv, stderr

from LRNN import gen_model
from data.gen_patch import gen_patch

def degrade(img, p):
    r, c = img.shape[:2]
    mask = np.random.choice([True, False], size=(r, c), p=[1-p, p])
    out = img.copy()
    out[mask] = 0
    return out

def datagen(pathnames, patchsize, batch_size):
    images = np.empty((batch_size, patchsize, patchsize, 3), dtype=np.float64)
    degrade_images = np.empty((batch_size, patchsize, patchsize, 4), dtype=np.float64)
    batch_size = batch_size // 1
    while True:
        np.random.shuffle(pathnames)
        for i in range(0, len(pathnames), batch_size):
            n_sample = 0
            for f in pathnames[i:i+batch_size]:
                img = skimage.io.imread(f)
                for _ in range(1):
                    patch = gen_patch(img, patchsize)
                    patch = skimage.util.img_as_float(patch)
                    if patch.ndim == 2:
                        patch = skimage.color.gray2rgb(patch)
                    images[n_sample] = patch
                    luminance_ch = rgb2yuv(patch)[:,:,0]
                    degrade_images[n_sample] = np.concatenate(
                        [degrade(patch, 0.05), np.expand_dims(luminance_ch, -1)],
                        axis=-1
                    )
                    n_sample = n_sample + 1
            yield (degrade_images[:n_sample], images[:n_sample])

start_from_iter = 0
if len(argv) < 2:
    raise ValueError("Usage: {} [cascade | parallel] [start iter]".format(argv[0]))

if len(argv) > 2:
    start_from_iter = int(argv[2])
    if start_from_iter <= 0:
        raise ValueError("start_from_iter should be larger than 0")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0x5EED)

batch_size = 32
patchsize = 96
n_epoch = 30

model = gen_model((patchsize, patchsize, 4), argv[1])
model.compile(optimizer='Adamax', loss=Keras.losses.mse)
# Keras.utils.plot_model(model, to_file='model.png')

datadir = './data/train2014'
pathnames = [os.path.join(datadir, f) for f in os.listdir(datadir)]
n_sample = len(pathnames)
steps_per_epoch=200
n_epoch=300000

valdir = './data/test2014'
valnames = [os.path.join(valdir, f) for f in os.listdir(valdir)]

outputdir = './model-color'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

if start_from_iter > 0:
    model.load_weights(
        os.path.join(outputdir, '%s-color-weights-%d.h5' % (argv[1], start_from_iter))
    )

for epoch in range(start_from_iter, n_epoch):
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
        os.path.join(outputdir, '%s-color-weights-%d.h5' % (argv[1], epoch+1))
    )
