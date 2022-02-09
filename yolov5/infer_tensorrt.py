from infer import check_img_size,letterbox,non_max_suppression,scale_coords
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size # 非动态输入
        # size = trt.volume(engine.get_binding_shape(binding))                       # 动态输入
        size = abs(size)                                # 上面得到的size(0)可能为负数，会导致OOM
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes) # cuda分配空间
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem)) # binding在计算图中的缓冲地址
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # 如果创建network时显式指定了batchsize，使用execute_async_v2, 否则使用execute_async
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

if __name__ == "__main__":
    #input
    image_size = 640
    conf_thres = 0.466
    iou_thres = 0.6
    batch_size = 1
    img0 = cv2.imread("../images/coco_1.jpg")
    weights = "weights/yolov5s.engine" 

    # 1.创建cudaEngine
    logger = trt.Logger(trt.Logger.INFO)
    with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # 2.将引擎应用到不同的GPU上配置执行环境
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # 3.推理
    #n6,s6,m6,l6,x6 stride=64
    #n,s,m,l,x stride=32
    stride=32
    imgsz = check_img_size(image_size, s=stride)

    #check imgsize
    tmp_shape = imgsz
    if isinstance(imgsz, int):
        tmp_shape = [imgsz,imgsz]
 
    # Padded resize
    img = letterbox(img0, new_shape=imgsz,auto=False)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]
    inputs[0].host = img.reshape(-1)

    # 如果是动态输入，需以下设置
    # context.set_binding_shape(0, img.shape)
    
    trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # 取得yolov5输出
    result_output = trt_outputs[3]
    # 由于tensorrt输出为一维向量，需要reshape到指定尺寸
    output_shape = (batch_size, -1, 85)
    pred = postprocess_the_outputs(result_output, output_shape)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                   agnostic=False)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    output=pred[0]
    # top_left_bottom_right
    boxes = output[:, :4]
    scores = output[:, 4]
    classes = output[:, 5]

    for box,score in zip(boxes,scores):
        top_left, bottom_right = box[:2].astype(np.int64).tolist(), box[2:4].astype(np.int64).tolist()
        result = cv2.rectangle(
            img0, tuple(top_left), tuple(bottom_right), tuple((255,0,0)), 2
        )
    cv2.imwrite("result.jpg",result)
