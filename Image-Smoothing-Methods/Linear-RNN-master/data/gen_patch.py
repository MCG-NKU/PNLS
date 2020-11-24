import numpy as np
import skimage
from skimage.transform import rescale, resize, rotate

__all__ = ['gen_patch']

def gen_patch(img, patchsize=96):
    r, c = img.shape[:2]
    #angle = (2 * np.random.random() - 1) * 15.0
    #img = rotate(img, angle)
    rs = int(np.random.random() * (r - patchsize / 2))
    cs = int(np.random.random() * (c - patchsize / 2))
    patch = img[rs:rs+patchsize, cs:cs+patchsize]
    r, c = patch.shape[:2]
    if min(r, c) < patchsize:
        patch = resize(patch, (patchsize, patchsize), mode='constant')
    return patch
