#!/usr/bin/python3
from contextlib import contextmanager
import time

@contextmanager
def Timer(name):
    start = time.time()
    yield
    finish = time.time()
    print(f'{name} took {finish-start:.4f} sec')

import pygame as pg
import numpy as np
import cv2

IW = 1920
IH = 1080
OW = IW // 2
OH = IH // 2

repeat = 10

isurf = pg.Surface((IW,IH), pg.SRCALPHA)
with Timer('pg.Surface with smoothscale'):
    for i in range(repeat):
        pg.transform.smoothscale(isurf, (OW,OH))

def cv2_resize(image):
    return cv2.resize(image, (OH,OW), interpolation=cv2.INTER_AREA)

i1 = np.zeros((IW,IH,3), np.uint8)
with Timer('np.zeros with cv2'):
    for i in range(repeat):
        o1 = cv2_resize(i1) 

i2 = pg.surfarray.pixels3d(isurf)
with Timer('pixels3d with cv2'):
    for i in range(repeat):
        o2 = cv2_resize(i2)

print('i1==i2 is',np.all(i1==i2))
print('o1==o2 is',np.all(o1==o2))
print('input shapes',i1.shape,i2.shape)
print('input types',i1.dtype,i2.dtype)
print('output shapes',o1.shape,o2.shape)
print('output types',o1.dtype,o2.dtype)

print('input strides',i1.strides,i2.strides)
print('output strides',o1.strides,o2.strides)

i3 = np.ndarray((IW,IH,3), np.uint8,
        strides=(4, IW*4, -1),
        buffer=b'\0'*(IW*IH*4),
        offset=2)

with Timer('pixels3d-like layout with cv2'):
    for i in range(repeat):
        o2 = cv2_resize(i3)

i4 = np.empty(i2.shape, i2.dtype)
with Timer('pixels3d-like copied to same-shape array'):
    for i in range(repeat):
        i4[:] = i2

with Timer('pixels3d-like to same-shape array, copyto'):
    for i in range(repeat):
        np.copyto(i4, i2)

import ctypes

def arr_params(surface):
    pixels = pg.surfarray.pixels3d(surface)
    width, height, depth = pixels.shape
    assert depth == 3
    xstride, ystride, zstride = pixels.strides
    oft = 0
    bgr = 0
    if zstride == -1: # BGR image - walk back 2 bytes
        # to get to the first blue pixel (this is good
        # for functions which don't care if it's BGR
        # or RGB as long as it's consistent)
        oft = -2
        zstride = 1
        bgr = 1
    assert xstride == 4
    assert zstride == 1
    assert ystride == width*4
    base = pixels.ctypes.data_as(ctypes.c_void_p)
    ptr = ctypes.c_void_p(base.value + oft)
    return ptr, width, height, bgr

def rgba_buffer(p, w, h):
    type = ctypes.c_uint8 * (w * h * 4)
    return ctypes.cast(p, ctypes.POINTER(type)).contents

def cv2_resize_surface(src, dst):
    iptr, iw, ih, ibgr = arr_params(src)
    optr, ow, oh, obgr = arr_params(dst)
    assert ibgr == obgr

    ibuf = rgba_buffer(iptr, iw, ih)

    # reinterpret the array as RGBA height x width
    # (this "transposes" the image and possibly
    # flips R and B channels, in order to fit the data
    # into the layout cv2 expects)
    #
    # numpy's default strides are height*4, 4, 1
    iarr = np.ndarray((ih,iw,4), np.uint8, buffer=ibuf)
    
    obuf = rgba_buffer(optr, ow, oh)

    oarr = np.ndarray((oh,ow,4), np.uint8, buffer=obuf)

    cv2.resize(iarr, (ow,oh), oarr, interpolation=cv2.INTER_AREA)

osurf = pg.Surface((OW,OH), pg.SRCALPHA)
with Timer('attached RGBA with cv2'):
    for i in range(repeat):
        cv2_resize_surface(isurf, osurf)

i6 = np.zeros((IW,IH,4), np.uint8)
with Timer('np.zeros RGBA with cv2'):
    for i in range(repeat):
        o6 = cv2_resize(i6) 



image = np.zeros((IW,IH,3),dtype=np.uint8)
for y in range(IH):
    for x in range(IW):
        image[x,y,0] = 255*x//IW
        image[x,y,1] = 255*y//IH

import imageio
imageio.imsave('test.png',image)
imageio.imsave('test-resized.png',cv2_resize(image))

i2[:] = image
a=pg.surfarray.pixels_alpha(isurf)
a[:] = 255

with Timer('attached RGBA with cv2 [non-zero data]'):
    for i in range(repeat):
        cv2_resize_surface(isurf, osurf)

imageio.imsave('test-attached-resized.png',pg.surfarray.pixels3d(osurf))
pg.image.save(osurf, 'test-surface-resized.png')

def surface_arrs(src, dst):
    iptr, iw, ih, ibgr = arr_params(src)
    optr, ow, oh, obgr = arr_params(dst)
    assert ibgr == obgr

    ibuf = rgba_buffer(iptr, iw, ih)
    iarr = np.ndarray((ih,iw,4), np.uint8, buffer=ibuf)
    obuf = rgba_buffer(optr, ow, oh)
    oarr = np.ndarray((oh,ow,4), np.uint8, buffer=obuf)

    return iarr, oarr

def compare_perf(src, dst, func, name):
    src3d = pg.surfarray.pixels3d(src)
    dst3d = pg.surfarray.pixels3d(dst)

    dst3d[:] = 0
    with Timer(f'{name} with pixels3d'):
        for i in range(repeat):
            func(src3d, dst3d)

    val3d = dst3d.copy()

    src_arr, dst_arr = surface_arrs(src, dst)

    dst3d[:] = 0
    with Timer(f'{name} with default strides'):
        for i in range(repeat):
            func(src_arr, dst_arr)

    assert np.all(val3d == pg.surfarray.pixels3d(dst))

def assign(src, dst):
    dst[:] = src

def blur(src, dst):
    if dst.strides[-1] > 0: # otherwise cv2.GaussianBlur refuses to accept
      # the output image parameter
      cv2.GaussianBlur(src, (3,3), 0, dst)
    else:
      dst[:] = cv2.GaussianBlur(src, (3,3), 0)

def median_blur(src, dst):
    if dst.strides[-1] > 0: # otherwise cv2.GaussianBlur refuses to accept
      # the output image parameter
      cv2.medianBlur(src, 3, dst)
    else:
      dst[:] = cv2.medianBlur(src, 3, 0)

def invert(src, dst):
    dst[:] = 255 - src

def warp(src, dst):
    w, h = dst.shape[:2]
    M = np.float32([[1,0,100],[0,1,100]])
    dst[:] = cv2.warpAffine(src,M,(h,w))

pg.surfarray.pixels3d(isurf)[:] = image
osurf2 = pg.Surface((IW,IH), pg.SRCALPHA)
compare_perf(isurf, osurf2, warp, 'warp')
compare_perf(isurf, osurf2, assign, 'assign')
compare_perf(isurf, osurf2, invert, 'invert')
compare_perf(isurf, osurf2, blur, 'blur')
compare_perf(isurf, osurf2, median_blur, 'median_blur')
