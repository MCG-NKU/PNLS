import numpy as np
import keras.keras as Keras
import skimage
import skimage.io
from skimage.util import random_noise
from skimage.color import rgb2yuv
import matplotlib.pyplot as plt
import os
from sys import argv, stderr

from LRNN import gen_model

def die(msg):
    print(msg, file=stderr)
    exit(1)

def degrade(img, p):
    r, c = img.shape[:2]
    mask = np.random.choice([True, False], size=(r, c), p=[1-p, p])
    out = img.copy()
    out[mask] = 0
    return out


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0x5EED)

if len(argv) < 4:
    die('Usage: {} [model path] [output path] [cascade | parallel]'.format(argv[0]))

model = gen_model((None, None, 4), argv[3])
model.load_weights(argv[1])

datadir = './data/test-denoise'
filenames = os.listdir(datadir)

outputdir = argv[2]
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

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
    luminance_ch = rgb2yuv(img)[:,:,0]
    degrade_img = degrade(img, 0.05)
    input_img = np.concatenate(
        [degrade_img, np.expand_dims(luminance_ch, -1)],
        axis=-1
    )
    restore_img = model.predict(np.array([input_img]))[0]
    fname = os.path.splitext(filename)[0]
    skimage.io.imsave(os.path.join(outputdir, fname + '.png'), img)
    skimage.io.imsave(os.path.join(outputdir, fname + '.degrade.png'), degrade_img)
    skimage.io.imsave(os.path.join(outputdir, fname + '.restore.png'), restore_img)
