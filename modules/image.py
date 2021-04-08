import cv2
import numpy as np


def scale_and_clip(x, scale=255., bias=0., clip=(0., 255.)):
    return np.clip((x + bias) * scale, *clip)


def imsave(fp, img, normalized=False, shape=None):
    if normalized:
        img = img * 255.

    if shape is not None:
        if img.shape[0] > shape[0] and img.shape[1] > shape[1]:
            img = cv2.resize(img, shape)

    cv2.imwrite(fp, img)


def get_3ch_scaled_imgs(images, scale=255., bias=0.):
    images = scale_and_clip(images, scale, bias)
    assert images.ndim > 2
    assert images.shape[-1] in [1, 3]
    if images.ndim == 3:
        images = np.expand_dims(images, 0)
    if images.shape[-1] == 1:
        images = mono_to_3ch_imgs(images)
    return images


def mono_to_3ch_imgs(imgs):
    return np.tile(imgs, [1, 1, 1, 3])


def get_tiles_of_array(array, shape=None):
    nsamples = len(array)
    if shape is None:
        assert np.prod(nsamples) ** 0.5 % 1 == 0.
        shape = [int(np.prod(nsamples) ** 0.5)] * 2
    else:
        assert np.prod(shape) >= nsamples
        if np.prod(shape) > nsamples:
            nblanks = np.prod(shape) - nsamples
            blanks = np.ones([nblanks, *array.shape[1:]]) * 255.
            array = np.vstack([array, blanks])
            nsamples = len(array)

    ii = np.arange(nsamples).reshape(shape)
    return np.concatenate([np.concatenate(x, axis=1) for x in [array[i] for i in ii]], axis=0)
