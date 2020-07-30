#!/usr/bin/env python3
'''
This file implements an API wrapper for parts of the OpenCV CUDA module so that
code can interchangeably use the OpenCL or CUDA backends.
'''
import functools
import os

import cv2


def num_cuda_devices():
    ordinals = os.environ.get('GPU_DEVICE_ORDINAL', '').split(',')
    return len([o for o in ordinals if o != ''])


def get_accelerator(n):
    if n < num_cuda_devices():
        cv2.cuda.setDevice(n)
        return OpenCV_CUDA()
    return OpenCV_CPU()


class OpenCV_CPU:
    is_gpu = False


    def push_stream(self):
        pass
    
    def await_stream(self):
        pass

    def get_stream(self):
        raise NotImplemented()
    
    def pop_stream(self):
        pass
    

    def upload(self, o):
        return o

    def download(self, o):
        return o


    def __getattr__(self, name):
        return getattr(cv2, name)


class OpenCV_OpenCL(OpenCV_CPU):
    is_gpu = False  # maybe?

    def upload(self, o):
        return cv2.UMat(o)

    def download(self, o):
        return o.get()  # return as np.ndarray


class OpenCV_CUDA:
    is_gpu = True

    def __init__(self):
        self.__streams__ = []

    def __del__(self):
        while self.__streams__:
            self.pop_stream()


    def push_stream(self):
        self.__streams__.append(cv2.cuda_Stream())
    
    def await_stream(self):
        self.__streams__[-1].waitForCompletion()

    def get_stream(self):
        return self.__streams__[-1]
    
    def pop_stream(self):
        self.await_stream()
        self.__streams__.pop()


    def upload(self, o):
        return cv2.cuda_GpuMat(o)

    def download(self, o):
        return o.download(stream=self.get_stream())


    # Has a different name but compatible interface
    FarnebackOpticalFlow = cv2.cuda_FarnebackOpticalFlow


    # The cv2.cuda.merge() function does not allocate space for the result
    # automatically, instead it returns np.ndarray?
    def merge(self, channels):
        assert len(channels) == 3
        output = cv2.cuda_GpuMat(channels[0].size(), cv2.CV_8UC3)
        cv2.cuda.merge(channels, output)
        return output


    # Super hacky, but this trick lets us figure out if we can pass a CUDA
    # stream to the function
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def __takes_stream(name):
        docstring = getattr(getattr(cv2.cuda, name), '__doc__', '')
        return 'stream' in docstring


    def __getattr__(self, name):
        value = getattr(cv2.cuda, name)
        if callable(value) and self.__takes_stream(name) and self.__streams__:
            return functools.partial(value, stream=self.get_stream())
        return value
