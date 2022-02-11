# encoding: utf-8
"""
@author:  lishuang
@contact: qqlishuang@gmail.com
"""
# For our custom calibrator
from calibrator import Yolov5EntropyCalibrator

from build_engine import build_engine



def main():
    fp16_mode = False
    int8_mode = True
    print('*** onnx to tensorrt begin ***')
    onnx_file_path = "weights/yolov5s.onnx"
    engine_file_path = "weights/yolov5s.engine"

    calib=None
    if int8_mode:
        # image size
        height = 640
        width = 640
        test_set = "coco128/images/train2017"
        engine_file_path = "weights/yolov5s_int8.engine"
        # Now we create a calibrator and give it the location of our calibration data.
        # We also allow it to cache calibration data for faster engine building.
        calibration_cache = "weights/yolov5_calibration.cache"
        calib = Yolov5EntropyCalibrator(test_set, cache_file=calibration_cache,height=height,width=width)

    batch_size=1
    engine = build_engine(onnx_file_path, batch_size,fp16_mode=fp16_mode, int8_mode=int8_mode,calib=calib)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

if __name__ == '__main__':
    main()
