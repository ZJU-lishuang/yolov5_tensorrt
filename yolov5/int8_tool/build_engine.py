# encoding: utf-8
"""
@author:  lishuang
@contact: qqlishuang@gmail.com
"""
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

# This function builds an engine from a onnx model.
def build_engine(onnx_file_path, batch_size=32,fp16_mode=False, int8_mode=False, calib=None):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config,trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        config.max_workspace_size = GiB(1)
        if fp16_mode:
            if not builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        elif int8_mode:
            if not builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib
        # Parse onnx model
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
            assert network.num_layers > 0, 'Failed to parse ONNX model. \
                        Please check if the ONNX model is compatible '
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

        # Build engine and do int8 calibration.
        return builder.build_engine(network, config)