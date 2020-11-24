import numpy as np
import keras.keras as Keras
import skimage
import skimage.io
from skimage.util import random_noise
import matplotlib.pyplot as plt
import os
from sys import argv, stderr

from LRNN import gen_model

def die(msg):
    print(msg, file=stderr)
    exit(1)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0x5EED)

if len(argv) < 4:
    die('Usage: {} [model path] [output path] [Gaussian noise variance]'.format(argv[0]))

model = gen_model((None, None, 3), 'parallel')
model.load_weights(argv[1])

datadir = './data/test-denoise'
filenames = os.listdir(datadir)

outputdir = argv[2]
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

gauss_var = float(argv[3])

for filename in filenames:
    pathname = os.path.join(datadir, filename)
    img = skimage.io.imread(pathname)
    r, c = img.shape[:2]
    if (r % 16 != 0) or (c % 16 != 0):
        print('Image dimension is not 16-divisible: ' + filename)
        continue
    img = skimage.img_as_float(img)
    if img.ndim == 2:
        img = skimage.color.gray2rgb(img)
    noisy_img = random_noise(img, var=gauss_var)
    denoise_img = model.predict(np.array([noisy_img]))
    denoise_img = denoise_img[0]
    fname = os.path.splitext(filename)[0]
    skimage.io.imsave(os.path.join(outputdir, fname + '.png'), img)
    skimage.io.imsave(os.path.join(outputdir, fname + '.noisy.png'), noisy_img)
    skimage.io.imsave(os.path.join(outputdir, fname + '.denoise.png'), denoise_img)
