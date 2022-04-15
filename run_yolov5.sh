sudo docker run --gpus "device=0" -it --rm --net=host --shm-size=1g --ipc=host \
-v$(pwd)/:/workspace/yolov5_tensorrt \
-v/home/ls/softwares/TensorRT:/workspace/TensorRT \
-v/home/ls/softwares/cudnn:/workspace/cudnn \
-v/home/ls/softwares/opencv-4.5.5:/workspace/opencv \
-w /workspace/yolov5_tensorrt nvcr.io/nvidia/pytorch:22.03-py3