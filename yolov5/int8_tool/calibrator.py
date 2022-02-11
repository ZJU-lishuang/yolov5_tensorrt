# encoding: utf-8
"""
@author:  lishuang
@contact: qqlishuang@gmail.com
"""
import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import glob
import cv2
from pathlib import Path
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def preprocess_image(image_raw,height,width):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    #image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    #image = np.ascontiguousarray(image)
    #return image,image_raw,h,w
    return image

class DataLoader:
    def __init__(self,path,batch_size,height,width):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        BATCH_SIZE = batch_size

        self.index = 0
        self.length = int(len(images)/batch_size)
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        # self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        self.img_list = images
        # assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(path) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32)
        self.height = height
        self.width = width

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess_image(img,self.height,self.width)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

class Yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, height=640,width=640,batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = DataLoader(training_data,batch_size,height,width)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data.calibration_data.nbytes)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        batch = self.data.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.device_input, batch)

        return [int(self.device_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
