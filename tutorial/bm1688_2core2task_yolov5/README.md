# bm1688_2core2task_yolov5例程

### 1.说明

这是一个在BM1688上运行双核双任务的例子，BM1688的TPU有两个npu core，“双核双任务”的意思就是在这两个npu core上分别跑一个bmodel。
该例程的重点在于指导用户如何使用BM1688的双核推理功能，没有后处理加速、前后处理/推理并行功能，如果您需要其他版本的教程，可以参考[YOLOv5例程](../../sample/YOLOv5/README.md#22-算法特性).

### 2.准备数据

可以通过如下命令下载测试视频和模型：
```bash
mkdir -p datasets
mkdir -p models/BM1688
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/common/test_car_person_1080P.mp4
python3 -m dfss --url=open@sophgo.com:sophon-demo/tutorials/bm1688_2core2task_yolov5/yolov5s_v6.1_3output_int8_4b.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/common/coco.names
mv test_car_person_1080P.mp4 datasets/
mv yolov5s_v6.1_3output_int8_4b.bmodel models/BM1688
mv coco.names datasets/
```
模型来源：[YOLOv5例程](../../sample/YOLOv5/README.md#3-准备模型与数据)

### 3.样例测试

- [C++例程](./cpp/README.md)
- [Python例程](./python/README.md)